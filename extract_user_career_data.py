import sys
import json
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from datetime import datetime

tokenizer = RegexpTokenizer('\w+')

distinct_days_seen = defaultdict(lambda: set())

for line in sys.stdin:
    dyad, acts = line.strip().split('\t')
    dyad = json.loads(dyad.replace("'",'"').replace(")","]").replace("(","["))
    acts = json.loads(acts)
    for act in acts:
        spin, i, timestamp, text = act
        ego = dyad[0] if spin == 1 else dyad[1]
        day = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        distinct_days_seen[ego].add(day)

for ego, days in distinct_days_seen.items():
    days = sorted(list(days))
    for day, i in zip(days, range(len(days))):
        print('%s\t%s\t%d' % (ego, day, i))
