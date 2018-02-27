import json
from torch.utils.data import Dataset


class DyadDataset(Dataset):
    
    def __init__(self):
        
        self.data = []

        with open('dyad_dataset.dev') as f:
            
            for l in f.readlines():
                
                dyad, acts = l.split('\t')
                dyad = json.loads(dyad.replace("'",'"').replace(")","]").replace("(","["))
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


