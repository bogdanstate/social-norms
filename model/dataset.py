import json
from torch.utils.data import Dataset
from config import DATA_PATH, DATASET_NAME
import torch

class DyadDataset(Dataset):
    
    def __init__(self):
        
        self.data = []
        self.dyad_lut = set()
        path = DATA_PATH + DATASET_NAME
        with open(path) as f:
            
            for l in f.readlines():
                
                # dyad - encodes information about the dyad
                # e.g.: ["noob", "graybeard"]
                # acts - encodes dialog acts - list of tuples, where each tuple
                # is an act.
                dyad, acts = l.split('\t')
                # super-hacky, need to change
                dyad = json.loads(dyad.replace("'",'"').replace(")","]").replace("(","["))
                # dyad = json.loads(dyad)
                dyad = tuple(dyad)

                self.dyad_lut.add(dyad)
                
                acts = json.loads(acts)
                concat_acts = []

                prev_act = None
                for act in acts:
                    if prev_act is None:
                        prev_act = act
                    else:
                        spin, i, timestamp, text = act
                        prev_spin, prev_i, prev_timestamp, prev_text = prev_act
                        if spin == prev_spin:
                            prev_act = (spin, prev_i, prev_timestamp, prev_text + text)
                        else:
                            concat_acts += [prev_act]
                            prev_act = act
                concat_acts += [prev_act]
                self.data += [(dyad, concat_acts)]

    def __getitem__(self, index):
        dyad, concat_acts = self.data[index]
        return (dyad, concat_acts)

    def __len__(self):
        return len(self.data)
    
    def get_dyad_lut(self):
        processed_lut = dict((k, v) for k, v in zip(list(self.dyad_lut), range(len(self.dyad_lut))))
        return processed_lut

