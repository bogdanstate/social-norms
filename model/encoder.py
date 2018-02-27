from torch.autograd import Variable
import torch

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

