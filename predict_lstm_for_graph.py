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
from model.config import DATA_PATH

EMBS_FILE = 'embs.pickle'
BATCH_SIZE = 1000
CUDA = True
NUM_EPOCHS = 20
NUM_ENTITIES = 287345
DIM = 100

with open('lstm_lstm.pickle', 'rb') as f:
    lstm = pickle.load(f).cpu()

with open('lstm_lstm.model', 'rb') as f:
    model = pickle.load(f).cpu()

with open('%s/embedding_model.vec' % DATA_PATH) as f:
    embs = []
    words_lut = {}
    i = 0
    for l in f.readlines()[1:]:
        if i <= NUM_ENTITIES:
            word, emb = l.strip().split(' ', 1)
            words_lut[i] = word
            i += 1

emb = Embedding(NUM_ENTITIES, DIM, padding_idx=0)
emb.load_state_dict(torch.load('emb_post_lstm.txt'))


if CUDA:
    lstm = lstm.cuda()
    emb = emb.cuda()
    model = model.cuda()

class DyadLSTMDataset(Dataset):
    def __init__(self):
        self.labels = []
        self.words = []
        self.user1 = []
        self.user2 = []
        self.text = []
        self.spin = []
        with open('dyad_dataset_for_lstm.txt') as f:
            for line in f.readlines():
                user1, user2, acts, pred1, num_days1, pred2, num_days2 = line.strip().split('\t')
                acts = json.loads(acts)
                pred1 = float(pred1)
                pred2 = float(pred2)
                labels = [[pred1, pred2] if a[0] == 1 else [pred2, pred1] for a in acts if len(a[3]) > 0]
                spin = [a[0] for a in acts if len(a[3]) > 0]
                words = [a[3] for a in acts if len(a[3]) > 0]
                text = [" ".join([words_lut[y] for y in x]) for x in words]
                self.labels += labels
                self.words += words
                self.user1 += [user1] * len(labels)
                self.user2 += [user2] * len(labels)
                self.text += text
                self.spin += spin
    def __getitem__(self, i):
        return (self.labels[i], self.words[i], self.user1[i], self.user2[i], self.text[i], self.spin[i])

    def __len__(self):
        return len(self.labels)

def collation(batches):
    return (
        [batch[0] for batch in batches],
        [batch[1] for batch in batches],
        [batch[2] for batch in batches],
        [batch[3] for batch in batches],
        [batch[4] for batch in batches],
        [batch[5] for batch in batches],
    ) 

dataset = DyadLSTMDataset()
idx = [x for x in range(dataset.__len__())]

loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collation, shuffle=False)

def compute_prediction(batch):
    labels, words, user1, user2, text, spin = batch
    lengths = [len(x) for x in words]
    max_len = max(lengths)
    z = zip(words, lengths, labels, user1, user2, text, spin)
    z = sorted(z, key=lambda x: x[1] * -1)
    words, lengths, labels, user1, user2, text, spin = zip(*z)
    
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
    return (out, labels, user1, user2, text, spin)

for batch in loader:
    out, labels, user1, user2, text, spin = compute_prediction(batch)
    out = out.cpu().data.numpy().tolist()
    labels = labels.cpu().data.numpy().tolist()
    out = ["%.4f\t%.4f" % tuple(x) for x in out]
    labels = ["%.4f\t%.4f" % tuple(x) for x in labels]
    to_print = zip(out, labels, user1, user2, spin, text)
    
    print("\n".join(["\t".join([str(y) for y in x]) for x in to_print]))
