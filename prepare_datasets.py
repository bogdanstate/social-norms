import sys
import json
from nltk.tokenize import RegexpTokenizer
import torch
import pickle

tokenizer = RegexpTokenizer('\w+')
with open('embedding_model.vec') as f:
    embs = []
    words = {}
    i = 0
    for l in f.readlines()[1:]:
        if i <= 1000:
            word, emb = l.strip().split(' ', 1)
            emb = emb.split(' ')
            emb = [float(x) for x in emb]
            embs += [emb]
            words[word] = i
            i += 1

embs = torch.FloatTensor(embs)
with open('embs1.pickle', 'wb') as f:
    pickle.dump(embs, f)
with open('words1.pickle', 'wb') as f:
    pickle.dump(words, f)

for line in sys.stdin:
    dyad, acts = line.strip().split('\t')
    acts = json.loads(acts)
    new_acts = []
    for act in acts:
        spin, i, timestamp, text = act
        tokens = tokenizer.tokenize(text)
        tokens = [words[x.lower()] for x in tokens if x.lower() in words]
        new_act = (spin, i, timestamp, tokens)
        new_acts += [new_act]
    print('%s\t%s' % (dyad, json.dumps(new_acts)))
