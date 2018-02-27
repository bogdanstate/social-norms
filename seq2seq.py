import torch
from torch.nn import Module, Embedding, Parameter
from torch.autograd import Variable
from dataset import DyadDataset
from encoder import DyadEncoder
from decoder import DyadDecoder
import json
import pickle

max_length = 20

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

SOS_token = 0
criterion = torch.nn.NLLLoss()
i = 0
for item in dataset:
    loss = 0
    dyad, acts = item
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
