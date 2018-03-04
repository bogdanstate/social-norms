import torch
from torch.nn import Module, Embedding, Parameter
from torch.autograd import Variable

class StatusEmb(Module):

    def __init__(self, dim, dyad_lut, cuda=False):
        
        super(StatusEmb, self).__init__()
        self.dyad_lut = dyad_lut
        self.dim = dim
        self.cuda = cuda

        num_entities = len(dyad_lut)

        self.emb = Embedding(num_entities, self.dim)
        if self.cuda:
            self.emb = self.emb.cuda()

    def forward(self, dyad):
        
        dyad = self.dyad_lut[dyad]
        dyad = torch.LongTensor([dyad])
        dyad = Variable(dyad)
        if self.cuda:
            dyad = dyad.cuda()
        out = self.emb(dyad)
        return out

