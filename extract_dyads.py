import os
import json
from datetime import datetime
from time import mktime
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool

files = os.listdir('processed_files')

def process_one_file(fname):
    dyads_acts = {}
    print(fname)
    with open('processed_files/' + fname) as f:
        year, month, day, _ = fname.split('-', 3)
        lines = f.readlines()
        for line in lines:
            dyad, acts = line.split('\t', 1)
            acts = json.loads(acts)
            processed_acts = []
            for act in acts:
                spin, line_no, minute, text = act
                if minute != "===":
                    try:
                        hour, minute = minute[1:-1].split(':')
                    except:
                        print(minute)
                    t = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute))
                    processed_act = (spin, line_no, mktime(t.timetuple()), text)
                    processed_acts += [processed_act]
            dyads_acts[dyad] = processed_acts
    return dyads_acts


fnames = files
jobs = []
pool = ThreadPool(12)
results = pool.map(process_one_file, fnames)
pool.close()
pool.join()

all_dyads_acts = defaultdict(lambda : [])
for dyads_acts in results:
    for k, v in dyads_acts.items():
        all_dyads_acts[k] += v

with open('dyads.txt', 'w') as f:
    for k, v in all_dyads_acts.items():
        f.write("%s\t%s\n" % (k, json.dumps(v)))
