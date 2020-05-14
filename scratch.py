import torch
import torch.nn as nn
import mytiming
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

torch.manual_seed(42)
i, f, j, o = torch.chunk(torch.arange(100*4*5, dtype=torch.float).view(100, 4*5)*5+100, 4, 1)
goal = torch.normal(torch.ones(100, 4, 5)+4)*7
ln = nn.LayerNorm((4, 5))
# ln = MyLayerNorm((4, 5))
viewed = torch.cat((i, f, j, o), 1).view(100, 4, 5)
lose_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(ln.parameters(), lr=1e-2)
out = viewed
for e in range(20000):
    optimizer.zero_grad()
    out = ln(viewed)
    loss = lose_fn(out, goal)
    loss.backward()
    optimizer.step()
    
print(out-goal)
print(lose_fn(out, goal))