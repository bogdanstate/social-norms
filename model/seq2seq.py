import torch
from torch.nn import Module, Embedding, Parameter
from torch.autograd import Variable

from dataset import DyadDataset
from encoder import DyadEncoder
from decoder import DyadDecoder
from status_emb import StatusEmb

import json
import pickle
from config import DATA_PATH, EMBS_FILE

max_length = 20

dataset = DyadDataset()
dyad_lut = dataset.get_dyad_lut()

embs_path = DATA_PATH + EMBS_FILE
cuda = True

with open(embs_path, 'rb') as f:
    emb_params = pickle.load(f)
num_entities, dim = emb_params.size()

emb = Embedding(num_entities, dim)
status_emb = StatusEmb(dim, dyad_lut, cuda)

emb.weight = Parameter(emb_params)

encoder = DyadEncoder(emb, dim, cuda=cuda)
decoder = DyadDecoder(emb, dim, num_entities, cuda=cuda)

encoder_optim = torch.optim.Adam(encoder.parameters(), lr=10e-4)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=10e-4)
status_optim = torch.optim.Adam(status_emb.parameters(), lr=10e-4)

SOS_token = 0
criterion = torch.nn.NLLLoss()
i = 0
for item in dataset:
    dyad, acts = item

    if len(acts) > 1:
        for this_act, next_act in zip(acts[:-1], acts[1:]):
           
            loss = 0
            do_grad_descent = False

            status_vec = status_emb(dyad)
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            status_optim.zero_grad()

            encoder_hidden = encoder.initHidden()
            spin, _, timestamp, tokens = this_act
            next_spin, _, next_timestamp, next_tokens = next_act

            for ei in range(len(tokens)):
                token = torch.LongTensor([[tokens[ei]]])
                token = Variable(token)
                token = token.cuda() if cuda else token
                encoder_output, encoder_hidden = encoder(token, encoder_hidden)

            decoder_input = Variable(torch.LongTensor([[SOS_token]]))
            decoder_input = decoder_input.cuda() if cuda else decoder_input
            decoder_hidden = encoder_hidden * status_vec

            target_variable = Variable(torch.LongTensor(next_tokens))
            target_variable = target_variable.cuda() if cuda else target_variable

            if len(next_tokens) > 0:
                for di in range(len(next_tokens)):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    decoder_input = decoder_input.cuda() if cuda else decoder_input
                    loss += criterion(decoder_output, target_variable[di])
                    do_grad_descent = True

            if do_grad_descent: 
                loss.backward()
                encoder_optim.step()
                decoder_optim.step()
                status_optim.step()

                print(loss * 1.0 / len(next_tokens))
                
                if i % 100 == 0:
                    print('Saving models.')
                    with open('encoder.pickle', 'wb') as f:
                        torch.save(encoder, f)
                    with open('decoder.pickle', 'wb') as f:
                        torch.save(decoder, f)
                    with open('status_emb.pickle', 'wb') as f:
                        torch.save(status_emb, f)

                i += 1
