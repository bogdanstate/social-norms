import torch
import pickle
from encoder import DyadEncoder
from decoder import DyadDecoder
from torch.autograd import Variable

with open('../ubuntu-corpus/words1.pickle', 'rb') as f:
    words = pickle.load(f)

with open('../ubuntu-corpus/encoder.pickle', 'rb') as f:
    encoder = torch.load(f)
with open('../ubuntu-corpus/decoder.pickle', 'rb') as f:
    decoder = torch.load(f)

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('\w+')
text = 'ubuntu is great'
tokens = tokenizer.tokenize(text)
tokens = [x.lower() for x in tokens]
tokens = [words[x] for x in tokens if x in words]
tokens = torch.LongTensor([tokens])
tokens = Variable(tokens)
print(encoder.GRU)
encoder_hidden = encoder.initHidden()
print(encoder_hidden)
print(tokens)
encoder_output = encoder(tokens, encoder_hidden)

print(encoder_output)
