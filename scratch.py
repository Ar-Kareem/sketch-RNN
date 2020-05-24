import torch
import torch.nn as nn
import numpy as np
import hyperLSTM
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


class MyLayerNorm(nn.Module):
    def __init__(self, input_dim):
        super(MyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))
        if True or use_bias:
            self.beta = nn.Parameter(torch.ones(input_dim))
            
    def forward(self, x):
        dims = (2)
        mean = x.mean(dim=dims, keepdim=True)
        x_shifted = x - mean
        var = torch.mean((x_shifted)**2, dim=dims, keepdim=True)
        inv_std = torch.rsqrt(var + 1e-5)
        output = self.gamma * (x_shifted) * inv_std
        if True or use_bias:
            output += self.beta
        return output
class layernormstack(nn.Module):
    def __init__(self):
        super(layernormstack, self).__init__()
        # self.ln1 = nn.LayerNorm((5))
        # self.ln2 = nn.LayerNorm((5))
        # self.ln3 = nn.LayerNorm((5))
        # self.ln4 = nn.LayerNorm((5))
        self.ln = nn.LayerNorm((4, 5))
    def forward(self, x):
        # i, f, j, o = x
        # return torch.cat((self.ln1(i), self.ln2(f), self.ln3(j), self.ln4(o)), 1).view(100, 4, 5)
        return self.ln(x)
def layer_norm_test():
    torch.manual_seed(42)
    i, f, j, o = torch.chunk(torch.arange(100*4*5, dtype=torch.float).view(100, 4*5)*5+100, 4, 1)
    goal = torch.normal(torch.ones(100, 4, 5)+4)*7

    layernormstack1 = layernormstack()
    # ln = MyLayerNorm((4, 5))
    viewed = torch.cat((i, f, j, o), 1).view(100, 4, 5)
    lose_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(layernormstack1.parameters(), lr=1e-2)
    out = viewed
    for e in range(30000):
        optimizer.zero_grad()
        # out = layernormstack1((i, f, j, o))
        out = layernormstack1(viewed)
        loss = lose_fn(out, goal)
        loss.backward()
        optimizer.step()

    print(out-goal)
    print(lose_fn(out, goal))


def layer_norm_comparison():
    class PLWrapper(pl.LightningModule):
        def __init__(self, model, in_, out_):
            super(PLWrapper, self).__init__()
            self.model = model
            self.proj = nn.Linear(in_, out_)

        def forward(self, x, state=None):
            out, state = self.model(x, state)
            return self.proj(out).unsqueeze(-1), state
            # return self.proj(x.squeeze()).unsqueeze(-1), None

        def training_step(self, train_batch, batch_idx):
            x, y = train_batch
            out, _ = self.forward(x)
            loss = self.lose_fn(out, y)
            logs = {'train_loss': loss}
            return {'loss': loss, 'log': logs}

        def validation_step(self, val_batch, batch_idx):
            x, y = val_batch
            out, _ = self.forward(x)
            loss = self.lose_fn(out, y)
            return {'val_loss': loss}

        def validation_epoch_end(self, outputs):
            # called at the end of the validation epoch
            # outputs is an array with what you returned in validation_step for each batch
            # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

        def lose_fn(self, y_hat, y):
            if torch.any(y_hat<0):
                y_hat = y_hat + (-1*y_hat.min().item())
            return torch.sqrt(F.mse_loss(torch.log(y_hat + 1), torch.log(y + 1)))

        # ---------------------
        # TRAINING SETUP
        # ---------------------

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-2)

        def prepare_data(self):
            self.dataset = TupleDataset(10, 3)
            train_len = int(len(self.dataset)*0.8)
            self.data_train = DataLoader(self.dataset, batch_size=train_len,
                                         sampler=torch.utils.data.SequentialSampler(range(train_len)))
            self.data_valid = DataLoader(self.dataset, batch_size=len(self.dataset)-train_len,
                                         sampler=torch.utils.data.SequentialSampler(range(train_len, len(self.dataset))))

        def train_dataloader(self):
            return self.data_train

        def val_dataloader(self):
            return self.data_valid

    class TupleDataset(Dataset):
        def __init__(self, x_len, y_len):
            data = np.genfromtxt('data/BTC.csv', delimiter=',', skip_header=True)
            data = data[:-1, 4]
            self.x = torch.tensor([data[i : i+x_len]
                                   for i in range(len(data) - x_len - y_len)]).float()
            self.y = torch.tensor([data[i+x_len : i+x_len+y_len]
                                   for i in range(len(data) - x_len - y_len)]).float()

        def __len__(self):
            return int(len(self.x))

        def __getitem__(self, idx):
            # self.x is of shape [datalen, seq_len]
            # output should be [seq_len, 1]
            return self.x[idx].unsqueeze(1), self.y[idx].unsqueeze(1)

    model = hyperLSTM.LSTM(1, 100, batch_first=True, layer_norm=True)
    wrapper = PLWrapper(model, 100, 3)
    # wrapper = PLWrapper(None, 10, 3)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(wrapper)


if __name__ == '__main__':
    layer_norm_comparison()

