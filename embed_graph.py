import json
import sys
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torch.nn import EmbeddingBag, BCEWithLogitsLoss
from sklearn import metrics
import numpy as np

dyads = set()
DIM = 100
NUM_EPOCHS = 10
BATCH_SIZE = 10e3
# REG_WEIGHT = 0
REG_WEIGHT = 10e-6
CUDA = True

for l in sys.stdin:
    
    dyad, acts = l.split('\t')
    dyad = json.loads(dyad.replace("'",'"').replace(")","]").replace("(","["))
    dyads.add((dyad[0], dyad[1]))

node_ids = set([x[0] for x in dyads] + [x[1] for x in dyads])
node_ids = dict(x for x in zip(node_ids, range(len(node_ids))))
dyads = [(node_ids[x[0]], node_ids[x[1]]) for x in dyads]

a = [x[0] for x in dyads]
random.shuffle(a)
b = [x[1] for x in dyads]
negatives = [tuple(x) for x in zip(a, b)]
num_randoms = 3 * len(negatives)
randoms = [(random.randint(0, len(node_ids)-1), random.randint(1, len(node_ids)-1)) for _ in range(0, num_randoms)] 


features = dyads + negatives + randoms
labels = [1] * len(dyads) + [0] * len(negatives) + [0] * num_randoms
indices = [x for x in range(len(labels))]

features = torch.LongTensor(features)
labels = torch.FloatTensor(labels)

dataset = TensorDataset(features, labels)

print('Shuffling indices')
random.shuffle(indices)
valid_indices = indices[:10000]
train_indices = indices[10001:]
print('Creating samplers')

train_dataset = TensorDataset(features[train_indices], labels[train_indices])
valid_dataset = TensorDataset(features[valid_indices], labels[valid_indices])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_entities = len(node_ids)

emb = EmbeddingBag(num_entities, DIM, mode='sum')
optim = torch.optim.Adam(emb.parameters(), lr=10e-4)

if CUDA:
    emb = emb.cuda()

rev_node_ids = dict((x[1], x[0]) for x in node_ids.items())

def compute_loss(batch):
    ids, labels = batch
    labels = torch.FloatTensor(labels.numpy().tolist())
    ids = Variable(ids)
    labels = Variable(labels)
    if CUDA:
        ids = ids.cuda()
        labels = labels.cuda()

    out = emb(ids).sum(dim=1)
    weight = torch.FloatTensor([1 if x == 0 else 4 for x in labels.cpu().data.numpy().tolist()])
    weight = weight.cuda() if CUDA else weight
    loss_fun = BCEWithLogitsLoss(weight=weight)
    loss = loss_fun(out, labels)
    reg = [x for x in emb.parameters()][0].pow(2).sum()
    loss = loss + REG_WEIGHT * reg
    return (loss, out, labels)

for i in range(NUM_EPOCHS):
    print('Training epoch %d' % i)
    for batch in train_loader:
        optim.zero_grad()
        loss, out, labels = compute_loss(batch)
        loss.backward()
        optim.step()
    
    batch_loss = 0
    batch_out = None
    batch_labels = None
    for batch in valid_loader:
        loss, out, labels = compute_loss(batch)
        batch_loss += loss
        if batch_out is None:
            batch_out = out.cpu().data.numpy()
            batch_labels = labels.cpu().data.numpy()
        else:
            batch_out = np.cat(batch_out, out.cpu.data.numpy())
            batch_labels = np.cat(batch_labels, labels.cpu.data.numpy())

    print(batch_loss)
    fpr, tpr, thresholds = metrics.roc_curve(batch_labels, batch_out)
    auc = metrics.auc(fpr, tpr)
    print(auc)

    emb_vals = [x for x in emb.parameters()][0].data.cpu().numpy().tolist()
    with open('graph_emb.txt', 'w') as f:
        for i in range(len(emb_vals)):
            f.write('%d\t%s\t%s\n' % (i, rev_node_ids[i], "\t".join(["%.6f" % x for x in emb_vals[i]])))

