import torch
import torch.nn as nn
from torch.nn import init


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, forget_bias=1.0, dropout=0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_bias = forget_bias

        self.wx = nn.Linear(input_dim, 4*hidden_dim, bias=False)
        self.wh = nn.Linear(hidden_dim, 4*hidden_dim, bias=True)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.wh.weight.data)
        init.orthogonal_(self.wx.weight.data)
        init.constant_(self.wh.bias.data, val=0)

    def forward(self, x, state):
        """
        :param x: input of shape (seq_len, batch, input_dim)
        :param state: tuple of shape (h, c) where h and c are vectores of length hidden_dim
        :return:
        """

        assert x.dim() == 3, 'Expected Input of shape (seq_len, batch, input_dim)'
        x_full_seq = x
        h, c = state
        for x in x_full_seq:
            gates_i = self.wx(x)
            gates_h = self.wh(h)

            i, f, j, o = torch.chunk(gates_i + gates_h, 4, 1)

            if self.dropout:
                g = self.dropout(torch.tanh(j))
            else:
                g = torch.tanh(j)
            f += self.forget_bias

            c = c * self.sigmoid(f) + self.sigmoid(i) * g
            h = torch.tanh(c) * self.sigmoid(o)
        return h, (h, c)
