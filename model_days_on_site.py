import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.nn import EmbeddingBag, BCEWithLogitsLoss
import math
from sklearn.metrics import r2_score

BATCH_SIZE = 1000
NUM_EPOCHS = 100
CUDA = True
REG_PARAM = 10e-7
LR = 10e-5

data = pd.read_csv('careers_emb.best', sep='\t')
labels = data['num_days']
features = data.iloc[:,2:102]

labels = labels.as_matrix()
labels = np.log(labels).tolist()
features = features.as_matrix().tolist()

idx = [x for x in range(len(labels))]
random.shuffle(idx)
valid_idx = idx[:10000]
train_idx = idx[10000:]
print(train_idx[1:100])

features = [x + [y * y for y in x] for x in features]

featurest = torch.FloatTensor(features)
labelst = torch.FloatTensor(labels)
dataset = TensorDataset(featurest, labelst)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

train_features = torch.FloatTensor([features[x] for x in train_idx])
train_labels = torch.FloatTensor([labels[x] for x in train_idx])
valid_features = torch.FloatTensor([features[x] for x in valid_idx])
valid_labels = torch.FloatTensor([labels[x] for x in valid_idx])
train_dataset = TensorDataset(train_features, train_labels)
valid_dataset = TensorDataset(valid_features, valid_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = torch.nn.Sequential(
    torch.nn.Linear(200, 200),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(200, 1),
    torch.nn.Softplus(),
)
model = model.cuda() if CUDA else model
optim = torch.optim.Adam(model.parameters(), lr=LR)

for i in range(NUM_EPOCHS):
    for batch in train_loader:
        optim.zero_grad()
        features, labels = batch
        features = Variable(features)
        labels = Variable(torch.FloatTensor(labels.cpu().numpy().tolist()))
        features = features.cuda() if CUDA else features
        labels = labels.cuda() if CUDA else labels
        out = model(features)
        labels = (labels + 1).log()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(out, labels)
        reg = torch.cat([x.abs().sum() for x in model.parameters()]).sum()
        loss = loss + reg * REG_PARAM
        loss.backward()
        optim.step()

    valid_labels = []
    valid_out = []
    for batch in valid_loader:
        features, labels = batch
        features = Variable(features)
        labels = Variable(torch.FloatTensor(labels.cpu().numpy().tolist()))
        features = features.cuda() if CUDA else features
        labels = labels.cuda() if CUDA else labels
        out = model(features)
        labels = (labels + 1).log()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(out, labels)
        valid_labels += labels.cpu().data.numpy().tolist()
        valid_out += out.cpu().data.numpy().tolist()

    r2 = r2_score(valid_labels, valid_out)
    print(r2)

all_out = []
for batch in loader:
    features, labels = batch
    features = Variable(features)
    labels = Variable(torch.FloatTensor(labels.cpu().numpy().tolist()))
    features = features.cuda() if CUDA else features
    labels = labels.cuda() if CUDA else labels
    out = model(features)
    all_out += out.cpu().data.numpy().tolist()

data['pred_score'] = [x[0] for x in all_out]
data.to_csv('graph_emb_shrunk.txt', sep='\t')
