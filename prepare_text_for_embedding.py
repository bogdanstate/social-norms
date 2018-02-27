import sys
import json
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('\w+')

for line in sys.stdin:
    dyad, acts = line.strip().split('\t')
    acts = json.loads(acts)
    for act in acts:
        spin, i, timestamp, text = act
        tokens = tokenizer.tokenize(text)
        tokens = [x.lower() for x in tokens]
        tokens = " ".join(tokens)
        print(tokens)
