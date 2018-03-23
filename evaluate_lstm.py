import sys
import json
from nltk.tokenize import RegexpTokenizer
import torch
import pickle
from collections import defaultdict
from model.config import DATA_PATH
import math
from datetime import datetime
from torch.nn import Module, Embedding, Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
DAY_BUCKET_LOG = 5

tokenizer = RegexpTokenizer('\w+')

EMBS_FILE = 'embs.pickle'
BATCH_SIZE = 1000
CUDA = True
NUM_EPOCHS = 20
NUM_ENTITIES = 287345
DIM = 100

with open('lstm_lstm.model', 'rb') as f:
    model = pickle.load(f).cpu()

with open('%s/embedding_model.vec' % DATA_PATH) as f:
    embs = []
    words = {}
    i = 0
    for l in f.readlines()[1:]:
        if i <= NUM_ENTITIES:
            word, emb = l.strip().split(' ', 1)
            words[word] = i
            i += 1

with open('lstm_lstm.pickle', 'rb') as f:
    lstm = pickle.load(f).cpu()

emb = Embedding(NUM_ENTITIES, DIM, padding_idx=0)
emb.load_state_dict(torch.load('emb_post_lstm.txt'))

SENTENCES = [
    "Ubuntu",
    "Fuck Linux",
    "I am a n00b",
    "n00b I am",
    "This is a great idea for the community.",
    "I wish new users would respect the rules.",
    "Thank you so much!",
    "Thank you so much! From all my noob heart!",
    "Are you sure you want to do that?",
    "Should I choose Ubuntu or Fedora?",
    "I am having problems with the NVIDIA drivers",
    "Please watch your language, this is a PG-13 environment",
    "This is not the appropriate place to ask this question",
    "Hello friends, I am having a lot of problems with my Linux",
    "Could you please clarify your question?",
    "Have you even searched the forum",
    "This question has been asked before a hundred times",
    "sudo rm -rf /",
    "I have no idea",
    "I am just as clueless as you are",
    "Not too sure about what you should do here.",
    "I was trying to compile the kernel the other day",
    "Whatever works.",
    "I am against immigration",
    "Windows users are terrible people",
    "I know Linus personally",
    "We're all a bunch of n00bs, lol",
    "lol wut?",
    "Back in the old days we wrote FORTRAN.",
    "C++12 is so much worse than C99.",
    "Our Lord and Savior, Richard Stallman.",
    "eixt",
    "BSD rules",
]
for sentence in SENTENCES:
    tokens = tokenizer.tokenize(sentence)
    tokens = [words[x.lower()] for x in tokens if x.lower() in words]
    l = len(tokens)
    tokens = torch.LongTensor([tokens])
    tokens = Variable(tokens)
    out = emb(tokens)
    out = pack_padded_sequence(out, [l], batch_first=True)
    out, (h, c) = lstm(out)
    out = h.sum(dim=0)
    out = model(out)
    out = out.t().squeeze()
    print("%.5f\t%.5f\t%s" % tuple(out.data.numpy().tolist() + [sentence]))
