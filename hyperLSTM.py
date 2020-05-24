import torch
import torch.nn as nn
from torch.nn import init


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, forget_bias=1.0, dropout=0, layer_norm=False, batch_first=False):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_bias = forget_bias
        self.batch_first = batch_first
        self.zero_state = None

        self.wx = nn.Linear(input_dim, 4*hidden_dim, bias=False)
        self.wh = nn.Linear(hidden_dim, 4*hidden_dim, bias=True)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if layer_norm:
            self.layer_norm_gates = nn.LayerNorm([4, hidden_dim])
            self.layer_norm_c = nn.LayerNorm([hidden_dim])
        else:
            self.layer_norm_gates = None
            self.layer_norm_c = None

        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.wh.weight.data)
        init.orthogonal_(self.wx.weight.data)
        init.constant_(self.wh.bias.data, val=0)
        if self.layer_norm_c is not None:
            self.layer_norm_c.reset_parameters()
        if self.layer_norm_gates is not None:
            self.layer_norm_gates.reset_parameters()

    def forward(self, x, state=None):
        """
        :param x: input of shape (seq_len, batch, input_dim)
        :param state: tuple of shape (h, c) where h and c are vectores of length hidden_dim
        :return:
        """
        if self.batch_first:
            assert x.dim() == 3, 'Expected Input of shape (batch, seq_len, input_dim), got ' + str(x.size())
            x = x.transpose(0, 1)
        else:
            assert x.dim() == 3, 'Expected Input of shape (seq_len, batch, input_dim), got ' + str(x.size())

        if state is None:
            if self.zero_state is None:
                self.zero_state = tuple(torch.zeros(self.hidden_dim, device=x.device) for _ in range(2))

            state = self.zero_state

        assert len(state) == 2, 'Expected state of short and long term memory'

        x_full_seq = x
        h, c = state
        for x in x_full_seq:
            gates_i = self.wx(x)
            gates_h = self.wh(h)
            gates = gates_i + gates_h

            if self.layer_norm_gates is not None:
                gates = gates.view(-1, 4, self.hidden_dim)
                gates = self.layer_norm_gates(gates)
                gates = gates.view(-1, 4 * self.hidden_dim)

            i, f, j, o = torch.chunk(gates, 4, 1)

            g = torch.tanh(j)
            if self.dropout is not None:
                g = self.dropout(g)

            f += self.forget_bias

            c = c * self.sigmoid(f) + self.sigmoid(i) * g
            if self.layer_norm_c is not None:
                c = self.layer_norm_c(c)

            h = torch.tanh(c) * self.sigmoid(o)
        return h, (h, c)


class HyperLSTM(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 forget_bias=1.0,
                 dropout=0,
                 layer_norm=False,
                 batch_first=False,
                 hyper_hidden_dim=256,
                 hyper_embedding_dim=32,
                 hyper_dropout=0):

        super(HyperLSTM, self).__init__()
        self.hyper_cell = LSTM(input_dim, hyper_hidden_dim,forget_bias=forget_bias, dropout=hyper_dropout,
                               layer_norm=layer_norm, batch_first=batch_first)
        self.hyper_embedding_dim = hyper_embedding_dim
        self.hidden_dim = hidden_dim
        self.forget_bias = forget_bias
        self.batch_first = batch_first
        self.zero_state = None

        # Equations (11)
        # hyper_out to embedding for x
        self.w_hz_ix = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        self.w_hz_fx = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        self.w_hz_jx = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        self.w_hz_ox = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        # hyper_out to embedding for h
        self.w_hz_ih = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        self.w_hz_fh = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        self.w_hz_jh = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        self.w_hz_oh = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=True)
        # hyper_out to embedding for bias
        self.w_hz_ib = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=False)
        self.w_hz_fb = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=False)
        self.w_hz_jb = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=False)
        self.w_hz_ob = nn.Linear(hyper_hidden_dim, hyper_embedding_dim, bias=False)

        # Equations (12)
        # embedding to weight scaling vectors for x
        self.w_zd_ix = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        self.w_zd_fx = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        self.w_zd_jx = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        self.w_zd_ox = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        # embedding to weight scaling vectors for h
        self.w_zd_ih = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        self.w_zd_fh = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        self.w_zd_jh = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        self.w_zd_oh = nn.Linear(hyper_embedding_dim, hidden_dim, bias=False)
        # embedding to bias
        self.w_zd_ib = nn.Linear(hyper_embedding_dim, hidden_dim, bias=True)
        self.w_zd_fb = nn.Linear(hyper_embedding_dim, hidden_dim, bias=True)
        self.w_zd_jb = nn.Linear(hyper_embedding_dim, hidden_dim, bias=True)
        self.w_zd_ob = nn.Linear(hyper_embedding_dim, hidden_dim, bias=True)

        # also part of Equations (12)
        self.wx = nn.Linear(input_dim, 4*hidden_dim, bias=False)
        self.wh = nn.Linear(hidden_dim, 4*hidden_dim, bias=False)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if layer_norm:
            self.layer_norm_gates = nn.LayerNorm([4, hidden_dim])
            self.layer_norm_c = nn.LayerNorm([hidden_dim])
        else:
            self.layer_norm_gates = None
            self.layer_norm_c = None

        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        # hyper_out to embedding for x/h -> weight: constant 0 | bias: constant 1
        init.constant_(self.w_hz_ix.weight.data, val=0)
        init.constant_(self.w_hz_fx.weight.data, val=0)
        init.constant_(self.w_hz_jx.weight.data, val=0)
        init.constant_(self.w_hz_ox.weight.data, val=0)
        init.constant_(self.w_hz_ih.weight.data, val=0)
        init.constant_(self.w_hz_fh.weight.data, val=0)
        init.constant_(self.w_hz_jh.weight.data, val=0)
        init.constant_(self.w_hz_oh.weight.data, val=0)
        init.constant_(self.w_hz_ix.bias.data, val=1)
        init.constant_(self.w_hz_fx.bias.data, val=1)
        init.constant_(self.w_hz_jx.bias.data, val=1)
        init.constant_(self.w_hz_ox.bias.data, val=1)
        init.constant_(self.w_hz_ih.bias.data, val=1)
        init.constant_(self.w_hz_fh.bias.data, val=1)
        init.constant_(self.w_hz_jh.bias.data, val=1)
        init.constant_(self.w_hz_oh.bias.data, val=1)

        # hyper_out to embedding for bias -> weight: gaussian sd 0.01
        init.normal_(self.w_hz_ib.weight.data, std=0.01)
        init.normal_(self.w_hz_fb.weight.data, std=0.01)
        init.normal_(self.w_hz_jb.weight.data, std=0.01)
        init.normal_(self.w_hz_ob.weight.data, std=0.01)

        # wh and wx -> weight: Orthogonal
        init.orthogonal_(self.wx.weight.data)
        init.orthogonal_(self.wh.weight.data)

        # embedding to weight scaling vectors for x/h -> weight: constant 0.1/Nz (Nz=hyper_embedding_dim)
        init.constant_(self.w_zd_ix.weight.data, val=0.1/self.hyper_embedding_dim)
        init.constant_(self.w_zd_fx.weight.data, val=0.1/self.hyper_embedding_dim)
        init.constant_(self.w_zd_jx.weight.data, val=0.1/self.hyper_embedding_dim)
        init.constant_(self.w_zd_ox.weight.data, val=0.1/self.hyper_embedding_dim)
        init.constant_(self.w_zd_ih.weight.data, val=0.1/self.hyper_embedding_dim)
        init.constant_(self.w_zd_fh.weight.data, val=0.1/self.hyper_embedding_dim)
        init.constant_(self.w_zd_jh.weight.data, val=0.1/self.hyper_embedding_dim)
        init.constant_(self.w_zd_oh.weight.data, val=0.1/self.hyper_embedding_dim)

        # embedding to bias -> weight: constant 0 | bias: constant 0
        init.constant_(self.w_zd_ib.weight.data, val=0)
        init.constant_(self.w_zd_fb.weight.data, val=0)
        init.constant_(self.w_zd_jb.weight.data, val=0)
        init.constant_(self.w_zd_ob.weight.data, val=0)
        init.constant_(self.w_zd_ib.bias.data, val=0)
        init.constant_(self.w_zd_fb.bias.data, val=0)
        init.constant_(self.w_zd_jb.bias.data, val=0)
        init.constant_(self.w_zd_ob.bias.data, val=0)

        if self.layer_norm_c is not None:
            self.layer_norm_c.reset_parameters()
        if self.layer_norm_gates is not None:
            self.layer_norm_gates.reset_parameters()

    def forward(self, x, state=None):
        """
        :param x: input of shape (seq_len, batch, input_dim)
        :param state: tuple of shape (h, c) where h and c are vectores of length hidden_dim
        :return:
        """
        if self.batch_first:
            assert x.dim() == 3, 'Expected Input of shape (batch, seq_len, input_dim), got ' + str(x.size())
            x = x.transpose(0, 1)
        else:
            assert x.dim() == 3, 'Expected Input of shape (seq_len, batch, input_dim), got ' + str(x.size())

        if state is None:
            if self.zero_state is None:
                lstm_zero_state = tuple(torch.zeros(self.hidden_dim, device=x.device) for _ in range(2))
                hyper_cell_zero_state = tuple(torch.zeros(self.hyper_embedding_dim, device=x.device) for _ in range(2))
                self.zero_state = (lstm_zero_state, hyper_cell_zero_state)
            state = self.zero_state

        assert len(state) == 2, 'Expected state both LSTM'
        assert len(state[0]) == 2, 'Expected state of short and long term memory of LSTM cell'
        assert len(state[1]) == 2, 'Expected state of short and long term memory of Hyper LSTM cell'

        x_full_seq = x
        (h, c), (hyper_h, hyper_c) = state

        for x in x_full_seq:

            hyper_out, (hyper_h, hyper_c) = self.hyper_cell(x, (hyper_h, hyper_c))
            # hyper_out to embeds to weight scaling vectors for x
            d_ix = self.w_zd_ix(self.w_hz_ix(hyper_out))
            d_fx = self.w_zd_ix(self.w_hz_ix(hyper_out))
            d_jx = self.w_zd_ix(self.w_hz_ix(hyper_out))
            d_ox = self.w_zd_ix(self.w_hz_ix(hyper_out))
            # hyper_out to embeds to weight scaling vectors for h
            d_ih = self.w_zd_ih(self.w_hz_ih(hyper_out))
            d_fh = self.w_zd_ih(self.w_hz_ih(hyper_out))
            d_jh = self.w_zd_ih(self.w_hz_ih(hyper_out))
            d_oh = self.w_zd_ih(self.w_hz_ih(hyper_out))
            # hyper_out to embeds to bias
            d_ib = self.w_zd_ib(self.w_hz_ib(hyper_out))
            d_fb = self.w_zd_ib(self.w_hz_ib(hyper_out))
            d_jb = self.w_zd_ib(self.w_hz_ib(hyper_out))
            d_ob = self.w_zd_ib(self.w_hz_ib(hyper_out))

            gates_x = self.wx(x)
            gates_h = self.wh(h)
            ix, fx, jx, ox = torch.chunk(gates_x, 4, 1)
            ih, fh, jh, oh = torch.chunk(gates_h, 4, 1)
            # scale the gates
            ix = ix * d_ix
            fx = fx * d_fx
            jx = jx * d_jx
            ox = ox * d_ox
            ih = ih * d_ih
            fh = fh * d_fh
            jh = jh * d_jh
            oh = oh * d_oh
            # calculate the final value for the gates
            i = ix + ih + d_ib
            f = fx + fh + d_fb
            j = jx + jh + d_jb
            o = ox + oh + d_ob

            if self.layer_norm_gates is not None:
                gates = torch.stack((i, f, j, o), dim=1)
                gates = self.layer_norm_gates(gates)
                gates = gates.view(-1, 4 * self.hidden_dim)
                i, f, j, o = torch.chunk(gates, 4, 1)

            g = torch.tanh(j)
            if self.dropout is not None:
                g = self.dropout(g)

            f += self.forget_bias

            c = c * self.sigmoid(f) + self.sigmoid(i) * g
            if self.layer_norm_c is not None:
                c = self.layer_norm_c(c)

            h = torch.tanh(c) * self.sigmoid(o)
            
        return h, ((h, c), (hyper_h, hyper_c))

