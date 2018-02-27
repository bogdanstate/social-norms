import torch
import pickle
from seq2seq import Encoder, Decoder

with open('../ubuntu-corpus/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
with open('../ubuntu-corpus/decoder.pickle', 'rb') as f:
    decoder = pickle.load(f)
