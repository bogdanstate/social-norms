import torch
import pickle
import json

with open('status_emb.pickle', 'rb') as f:
    status_emb = torch.load(f)

lut = status_emb.dyad_lut
data = [*status_emb.emb.parameters()][0]
data = data.data.cpu().numpy().tolist()

for dyad, vec in zip(lut.keys(), data):
    print("%s: %s" % ("%s -- %s" % dyad, json.dumps(vec)))
