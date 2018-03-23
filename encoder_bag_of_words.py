import torch
import sys
import pandas as pd
import json
from datetime import datetime
import pickle
import numpy as np
from torch.nn import EmbeddingBag, MSELoss
from torch.nn import Module, Embedding, Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
from sklearn.metrics import r2_score

EMBS_FILE = 'embs.pickle'
BATCH_SIZE = 100000
CUDA = True
NUM_EPOCHS = 20

with open(EMBS_FILE, 'rb') as f:
    emb_params = pickle.load(f)
num_entities, dim = emb_params.size()
emb = EmbeddingBag(num_entities, dim, mode='sum')
emb.weight = Parameter(emb_params)


data = []
model = torch.nn.Sequential(
    torch.nn.Linear(100, 100),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(100, 10),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 2),
    torch.nn.Softplus(),
)
if CUDA:
    model = model.cuda()
    emb = emb.cuda()

optim = torch.optim.Adam(model.parameters(), lr=10e-4)
emb_optim = torch.optim.Adam(emb.parameters(), lr=10e-4)


class DyadLSTMDataset(Dataset):
    def __init__(self):
        self.labels = []
        self.words = []
        with open('dyad_dataset_for_lstm.txt') as f:
            for line in f.readlines():
                user1, user2, acts, pred1, num_days1, pred2, num_days2 = line.strip().split('\t')
                acts = json.loads(acts)
                pred1 = float(pred1)
                pred2 = float(pred2)
                labels = [[pred1, pred2] if a[0] == 1 else [pred2, pred1] for a in acts if len(a[3]) > 0]
                words = [a[3] for a in acts if len(a[3]) > 0]
                self.labels += labels
                self.words += words
    def __getitem__(self, i):
        return (self.labels[i], self.words[i])

    def __len__(self):
        return len(self.labels)

def collation(batches):
    return (
        [batch[0] for batch in batches],
        [batch[1] for batch in batches],
    ) 

dataset = DyadLSTMDataset()
idx = [x for x in range(dataset.__len__())]
random.shuffle(idx)
valid_idx = idx[:10000]
train_idx = idx[10000:]
valid_sampler = SubsetRandomSampler(valid_idx)
train_sampler = SubsetRandomSampler(train_idx)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collation, sampler=train_sampler) 
valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collation, sampler=valid_sampler) 
loss_fn = MSELoss()

def compute_prediction(batch):
    labels, words = batch
    offsets = [0] + [len(x) for x in words]
    offsets = np.cumsum(offsets).tolist()
    offsets = [x for x in offsets][:-1]
    words = [y for x in words for y in x]
   
    offsets = torch.LongTensor(offsets)
    labels = torch.FloatTensor(labels)
    words = torch.LongTensor(words)
     
    words = Variable(words)
    offsets = Variable(offsets)
    labels = Variable(labels)
    if CUDA:
        words = words.cuda()
        offsets = offsets.cuda()
        labels = labels.cuda()

    out = emb(words, offsets)
    out = model(out)
    return (out, labels)

for epoch in range(NUM_EPOCHS):
    for batch in train_loader:
        optim.zero_grad()
        emb_optim.zero_grad()
        out, labels = compute_prediction(batch)
        loss = loss_fn(out, labels)
        loss.backward()
        optim.step()
        emb_optim.step()

    valid_out = []
    valid_labels = []
    for batch in valid_loader:
        out, labels = compute_prediction(batch)
        out = out.cpu().data.numpy().tolist()
        labels = labels.cpu().data.numpy().tolist()
        valid_out += out
        valid_labels += labels
    r2 = r2_score(labels, out, multioutput='raw_values')
    print(r2)
with open('emb_post_bag_of_words.pickle', 'wb') as f:
    pickle.dump(emb, f)
with open('model_bag_of_words.pickle', 'wb') as f:
    pickle.dump(model, f)

#    for a in acts:
#        spin, _, timestamp, words = a
#        loss = None
#        lbl = [pred1, pred2] if spin == 1 else [pred2, pred1]
#        lbl = torch.FloatTensor(lbl)
#        lbl = Variable(lbl)
#        for word in words:
#            word = torch.LongTensor([[word]])
#            word = Variable(word)
#            vec = emb(word)
#            out = model(vec)
#            if loss is None:
#                loss = loss_fn(out, lbl)
#            else:
#                loss = loss + loss_fn(out, lbl)
#        if loss is not None:
#            loss = loss / len(words)
#            loss.backward()
#            optim.step()
#            print(loss)
