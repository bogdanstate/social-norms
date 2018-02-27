from collections import Counter
import sys
import pickle

counts = Counter()
for line in sys.stdin:
    words = line.strip().split(' ')
    for word in words:
        counts[word] += 1

for k, v in counts.most_common():
    print("%s\t%d" % (k, v))
