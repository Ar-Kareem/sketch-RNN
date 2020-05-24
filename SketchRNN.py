import torch
import torch.nn as nn
from torch.nn import init
from hyperLSTM import LSTM, HyperLSTM

class SketchRNN(nn.Module):

    def __init__(self, encoder_hidden_dim, z_dim, dropout):
        super(SketchRNN, self).__init__()
        self.encoder_forward = LSTM(5, hidden_dim=encoder_hidden_dim,
                                    dropout=dropout, layer_norm=True, batch_first=True)
        self.encoder_backward = LSTM(5, hidden_dim=encoder_hidden_dim,
                                     dropout=dropout, layer_norm=True, batch_first=True)
        
        self.decoder_rnn = HyperLSTM(5, encoder_hidden_dim, dropout=dropout, layer_norm=True,
                                     batch_first=True, hyper_hidden_dim=256, hyper_embedding_dim=32)

        self.proj_mu = nn.Linear(encoder_hidden_dim*2, z_dim, bias=True)
        self.proj_sigma = nn.Linear(encoder_hidden_dim*2, z_dim, bias=True)

    def forward(self, input_seq):

        encoder_forward_h = self.encoder_forward(input_seq)
        encoder_backward_h = self.encoder_backward(input_seq.flip(-1))
        encoder_h = torch.cat((encoder_forward_h, encoder_backward_h), dim=-1)

        mu = self.proj_mu(encoder_h)
        sigma = self.proj_sigma(encoder_h)
        sigma = torch.exp(sigma/2)
        z = torch.normal(mean=mu, std=sigma)


