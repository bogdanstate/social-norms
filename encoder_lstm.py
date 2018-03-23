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
from torch.nn.utils.rnn import pack_padded_sequence

EMBS_FILE = 'embs.pickle'
BATCH_SIZE = 1000
CUDA = True
NUM_EPOCHS = 2

with open(EMBS_FILE, 'rb') as f:
    emb_params = pickle.load(f)
num_entities, dim = emb_params.size()
emb = Embedding(num_entities, dim, padding_idx=0)
emb.weight = Parameter(emb_params, requires_grad=True)


data = []
lstm = torch.nn.LSTM(100, 100, bidirectional=True)
model = torch.nn.Sequential(
    #torch.nn.Linear(10, 10),
    #torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(100, 30),
    torch.nn.LeakyReLU(),
    #torch.nn.Linear(30, 10),
    #torch.nn.LeakyReLU(),
    torch.nn.Linear(30, 2), 
    # torch.nn.Softplus(),
)
if CUDA:
    lstm = lstm.cuda()
    model = model.cuda()
    emb = emb.cuda()

# optim = torch.optim.Adam(lstm.parameters(), lr=10e-4)
optim = torch.optim.Adam([x for x in lstm.parameters()], lr=10e-4)
model_optim = torch.optim.Adam([x for x in model.parameters()], lr=10e-4)
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
    lengths = [len(x) for x in words]
    max_len = max(lengths)
    z = zip(words, lengths, labels)
    z = sorted(z, key=lambda x: x[1] * -1)
    words, lengths, labels = zip(*z)
    
    words = [x + [0] * (max_len - len(x)) for x in words]
    
    # words = [y for x in words for y in x]
   
    labels = torch.FloatTensor(labels)
    #lengths = torch.FloatTensor(lengths)
    words = torch.LongTensor(words)
     
    words = Variable(words)
    labels = Variable(labels)
    #lengths = Variable(lengths)
    if CUDA:
        words = words.cuda()
        labels = labels.cuda()
        #lengths = lengths.cuda()
    
    out = emb(words)
    seq = pack_padded_sequence(out, lengths, batch_first=True)
    # c, h = lstm(seq)
    _, (h, c) = lstm(seq)
    out = h.sum(dim=0)
    out = model(out)
    return (out, labels)

def eval_performance(f, epoch, i):
    valid_out = []
    valid_labels = []
    for batch in valid_loader:
        out, labels = compute_prediction(batch)
        out = out.cpu().data.numpy().tolist()
        labels = labels.cpu().data.numpy().tolist()
        valid_out += out
        valid_labels += labels
    print('R2 score')
    r2 = r2_score(valid_labels, valid_out, multioutput='raw_values')
    print(r2)
    f.write("%.5f\t%.5f\t%d\t%d\n" % tuple(r2.tolist() + [epoch, i]))


for epoch in range(NUM_EPOCHS):
    i = 0
    for batch in train_loader:
        optim.zero_grad()
        model_optim.zero_grad()
        emb_optim.zero_grad()
        out, labels = compute_prediction(batch)
        # loss = (out - labels).abs().sum()
        loss = loss_fn(out, labels)
        loss.backward()
        optim.step()
        model_optim.step()
        emb_optim.step()

        if i % 100 == 0:
            stats = open('stats.txt', 'a')
            eval_performance(stats, epoch, i)
            stats.close()
        if i % 1000 == 0:
            print('Saving')
            with open('lstm_lstm.pickle', 'wb') as f:
                pickle.dump(lstm, f)
            with open('lstm_lstm.model', 'wb') as f:
                pickle.dump(model, f)
            torch.save(emb.state_dict(), 'emb_post_lstm.txt')
            #with open('emb_post_lstm.pickle', 'wb') as f:
            #     pickle.dump(emb, f)
        i += 1

with open('lstm_lstm.pickle', 'wb') as f:
    pickle.dump(lstm, f)

stats.close()
#with open('lstm_sequential.pickle', 'wb') as f:
#    pickle.dump(model, f)

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
#            out = lstm(vec)
#            if loss is None:
#                loss = loss_fn(out, lbl)
#            else:
#                loss = loss + loss_fn(out, lbl)
#        if loss is not None:
#            loss = loss / len(words)
#            loss.backward()
#            optim.step()
#            print(loss)
