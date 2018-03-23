import sys
import json
from nltk.tokenize import RegexpTokenizer
import torch
import pickle
from collections import defaultdict
from model.config import DATA_PATH
import math
from datetime import datetime
DAY_BUCKET_LOG = 5

tokenizer = RegexpTokenizer('\w+')
with open('%s/embedding_model.vec' % DATA_PATH) as f:
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
with open('%s/embs1.pickle' % DATA_PATH, 'wb') as f:
    pickle.dump(embs, f)
with open('%s/words1.pickle' % DATA_PATH, 'wb') as f:
    pickle.dump(words, f)

days_seen_dict = defaultdict(lambda: {})
with open('%s/careers.txt' % DATA_PATH) as f:
    lines = f.readlines()
for l in lines:
    user, day, days_on_site = l.strip().split('\t')
    days_on_site = int(days_on_site)
    days_seen_dict[user][day] = math.floor(math.log(days_on_site + 1, DAY_BUCKET_LOG))

for line in sys.stdin:
    dyad, acts = line.strip().split('\t')
   
    acts = json.loads(acts)
    spin, _, timestamp, _ = acts[0]
    # super-hacky, need to change
    dyad = json.loads(dyad.replace("'",'"').replace(")","]").replace("(","["))
    u1, u2 = dyad
    day = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    try:
        d1 = days_seen_dict[u1][day]
        d2 = days_seen_dict[u2][day]
    # this shouldn't happen
    except:
        d1 = -1
        d2 = -1

    dyad = (d1, d2) if spin == 1 else (d2, d1)

    new_acts = []
    for act in acts:
        spin, i, timestamp, text = act
        tokens = tokenizer.tokenize(text)
        tokens = [words[x.lower()] for x in tokens if x.lower() in words]
        new_act = (spin, i, timestamp, tokens)
        new_acts += [new_act]
    print('%s\t%s' % (json.dumps(dyad), json.dumps(new_acts)))
