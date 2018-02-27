import sys
import json
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from datetime import datetime

tokenizer = RegexpTokenizer('\w+')
distinct_days_seen = defaultdict(lambda: {})
tokens_dict = defaultdict(lambda: [])

with open('careers.txt') as f:
    for l in f.readlines():
        user, day, i = l.strip().split('\t')
        i = int(i)
        distinct_days_seen[user][day] = i

for line in sys.stdin:
    dyad, acts = line.strip().split('\t')
    dyad = json.loads(dyad.replace("'",'"').replace(")","]").replace("(","["))
    acts = json.loads(acts)
    for act in acts:
        spin, i, timestamp, text = act
        ego = dyad[0] if spin == 1 else dyad[1]
        if not 'bot' in ego :
            tokens = tokenizer.tokenize(text)
            tokens = set([x.lower() for x in tokens])
            day = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

            num_prior_days_in_chat_room = distinct_days_seen[ego][day]
            for token in tokens:
                tokens_dict[token] += [num_prior_days_in_chat_room]

for k, v in tokens_dict.items():
    n = len(v)
    avg = sum(v) * 1.0 / n
    print("%s\t%d\t%.2f" % (k, n, avg))
