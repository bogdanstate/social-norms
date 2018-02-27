import sys
import json
from nltk.tokenize import RegexpTokenizer
from datetime import datetime

tokenizer = RegexpTokenizer('\w+')

for line in sys.stdin:
    dyad, acts = line.strip().split('\t')
    acts = json.loads(acts)
    timestamps = [x[2] for x in acts]
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    num_interactions = len(acts)
    days_interactions = [
        datetime.fromtimestamp(x).strftime('%Y-%m-%d')
        for x in timestamps
    ]
    distinct_days_interactions = len(set(days_interactions))
    dyad = json.loads(dyad.replace("'",'"').replace(")","]").replace("(","["))

    print('%s\t%s\t%d\t%d\t%d\t%d' % (
        dyad[0], dyad[1], num_interactions, min_timestamp,
        max_timestamp, distinct_days_interactions
    ))
