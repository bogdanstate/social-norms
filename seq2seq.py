import torch
from torch.nn import Module, Embedding, Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset
import json
import pickle

max_length = 20

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


# from http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class DyadEncoder(torch.nn.Module):
    def __init__(self, emb, dim, cuda=False):
        
        super(DyadEncoder, self).__init__()
        self.emb = emb
        self.dim = dim
        self.GRU = torch.nn.GRU(self.dim, self.dim)
        self.cuda = cuda
        if self.cuda:
            self.GRU = self.GRU.cuda()
            self.emb = self.emb.cuda()

    def forward(self, input, hidden):
        embedded = self.emb(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.GRU(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.dim))
        return result.cuda() if self.cuda else result

class DyadDecoder(torch.nn.Module):
    
    def __init__(self, emb, dim, output_size):
        
        super(DyadDecoder, self).__init__()
        self.emb = emb
        self.dim = dim
        self.output_size = output_size
        self.GRU = torch.nn.GRU(self.dim, self.dim)
        self.out = torch.nn.Linear(self.dim, self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        output = self.emb(input).view(1, 1, -1)
        output = torch.nn.functional.relu(output)
        output, hidden = self.GRU(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.dim))
        return result.cuda() if self.cuda else result


dataset = DyadDataset()

EMBS_FILE = 'embs.pickle'
cuda = False

with open(EMBS_FILE, 'rb') as f:
    emb_params = pickle.load(f)
num_entities, dim = emb_params.size()
emb = Embedding(num_entities, dim)
emb.weight = Parameter(emb_params)

encoder = DyadEncoder(emb, dim)
decoder = DyadDecoder(emb, dim, num_entities)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=10e-4)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=10e-4)

item = dataset[100]
dyad, acts = item

SOS_token = 0
criterion = torch.nn.NLLLoss()
i = 0
for item in dataset:
    loss = 0
    for this_act, next_act in zip(acts[:-1], acts[1:]):
       
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        encoder_hidden = encoder.initHidden()
        spin, _, timestamp, tokens = this_act
        next_spin, _, next_timestamp, next_tokens = next_act
        #encoder_outputs = Variable(torch.zeros(max_length, dim))
        #encoder_outputs = encoder_outputs.cuda() if cuda else encoder_outputs

        for ei in range(len(tokens)):
            token = torch.LongTensor([[tokens[ei]]])
            token = Variable(token)
            token = token.cuda() if cuda else token
            encoder_output, encoder_hidden = encoder(token, encoder_hidden)
            #encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if cuda else decoder_input
        decoder_hidden = encoder_hidden

        loss = 0
        target_variable = Variable(torch.LongTensor(next_tokens))
        for di in range(len(next_tokens)):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if cuda else decoder_input
            loss += criterion(decoder_output, target_variable[di])
        
    loss.backward()
    encoder_optim.step()
    decoder_optim.step()
    print(loss)
    
    if i % 100 == 0:
        print('Saving models.')
        with open('encoder.pickle', 'wb') as f:
            pickle.dump(encoder, f)
        with open('decoder.pickle', 'wb') as f:
            pickle.dump(decoder, f)

    i += 1
