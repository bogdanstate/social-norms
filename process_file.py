import sys
from collections import defaultdict

filename = sys.argv[1]
output_file = sys.argv[2]

def process_lines(f):
    lines = []
    for line in f.readlines():
        try:
            (minute, ego, text) = line.strip().split(' ', 2)
            lines += [(minute, ego, text)]
        except:
            pass
    return lines

def extract_users(lines):
    users = set()
    lowercased_users = set()

    processed_lines = []
    for line in lines:
        (minute, ego, text) = line
        ego = ego[1:-1]
        if len(ego) > 1:
            while len(ego) > 1 and ego[-1] == '_':
                ego = ego[:-1]
            users.add(ego)
            lowercased_users.add(ego.lower())
        processed_lines += [(minute, ego, text)]
    return users, processed_lines, lowercased_users

def extract_prefixed_user(text, sep, u, f=lambda x: x):
    
    if sep in text:
        try:
            potential_alter, split_text = text.split(sep)
            if f(potential_alter) in u:
                return potential_alter, split_text
        except:
            pass
    return None, text


def extract_potential_alter(text, users, lowercased_users):
   
    i = 0
    SEPS = [":", ",", " "]
    F = [lambda x: x, lambda x: x.lower()]
    for f, u in zip(F, [users, lowercased_users]):
        for sep in SEPS:
            potential_alter, text = extract_prefixed_user(text, sep, u, f)
            if potential_alter is not None:
                return potential_alter, text
    return (None, text)

def extract_dialog(lines, users, lowercased_users):
    ego_memory = {}
    alter_memory = defaultdict(lambda : set())
    dialog = set()
    i = 0
    for line in lines:
        i += 1
        (minute, ego, text) = line
        alter, text = extract_potential_alter(text, users, lowercased_users)
        text = text.strip()
        if alter is not None:
            if alter in alter_memory:
                for x in alter_memory[alter]:
                    x = (x[0], x[1], x[2], ego, x[4])
                    dialog.add(x)
                max_i = x[0]
                if i - max_i >= 100:
                    alter_memory.pop(alter, None)

            dialog.add((i, minute, ego, alter, text))
            ego_memory[ego] = alter
            ego_memory[alter] = ego
        elif ego in ego_memory:
            dialog.add((i, minute, ego, ego_memory[ego], text))
        else:
            alter_memory[ego].add((i, minute, ego, None, text))

    return dialog

def extract_dyads(dialog):
    
    dyads = defaultdict(lambda : list())
    for act in dialog:
        
        i, minute, ego, alter, text = act
        u1, u2, spin = (ego, alter, 1) if ego < alter else (alter, ego, -1)

        dyads[(u1, u2)] += [(spin, i, minute, text)]

    return dyads

with open(filename) as f:
    lines = process_lines(f)

users, processed_lines, lowercased_users = extract_users(lines)
dialog = extract_dialog(processed_lines, users, lowercased_users)
dyads = extract_dyads(dialog)

import json


def merge_adjacent_lines(lines):

    current_act = None
    merged_lines = []
    for l in lines:
        spin, i, minute, text = l
        if current_act is not None:
            if spin != current_act[0]:
                merged_lines += [current_act]
                current_act = [l]
            else:
                current_act += [l]
        else:
            current_act = [l]
    merged_lines += [current_act]

    return lines

with open(output_file, 'w') as f:
    for k, v in dyads.items():
        spins = set([x[0] for x in v])
        if len(spins) == 2:
            v = sorted(v, key=lambda x: x[1])
            v = merge_adjacent_lines(v)
            v = json.dumps(v)
            f.write("%s\t%s\n" % (k, v))
