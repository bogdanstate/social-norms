import json
from torch.utils.data import Dataset


class DyadDataset(Dataset):
    
    def __init__(self):
        
        self.data = []
        self.users_lut = set()
        with open('dyad_dataset.dev') as f:
            
            for l in f.readlines():
                
                # dyad - encodes information about the dyad
                # e.g.: ["noob", "graybeard"]
                # acts - encodes dialog acts - list of tuples, where each tuple
                # is an act.
                dyad, acts = l.split('\t')
                dyad = json.loads(dyad)
                
                self.users_lut.add(dyad)
                
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
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def get_users_lut(self):
        return dict((k, v) for k, v in zip(list(self.users_lut), range(len(self.users_lut)))
