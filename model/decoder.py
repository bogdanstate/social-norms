from torch.autograd import Variable
import torch

# from http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class DyadDecoder(torch.nn.Module):
    
    def __init__(self, emb, dim, output_size, cuda=False):
        
        super(DyadDecoder, self).__init__()
        self.cuda = cuda
        self.emb = emb
        self.dim = dim
        self.output_size = output_size
        self.GRU = torch.nn.GRU(self.dim, self.dim)
        self.out = torch.nn.Linear(self.dim, self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        if self.cuda:
            self.emb = self.emb.cuda()
            self.GRU = self.GRU.cuda()
            self.out = self.out.cuda()
            self.softmax = self.softmax.cuda()

    def forward(self, input, hidden):
        output = self.emb(input).view(1, 1, -1)
        output = torch.nn.functional.relu(output)
        output, hidden = self.GRU(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.dim))
        return result.cuda() if self.cuda else result


