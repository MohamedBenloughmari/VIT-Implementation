import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import einops
import torch.nn.functional as F
from einops import einsum
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_layers=None, input_dim=None):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)
        self.Wo = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (batch, n_patch, dim)
        dim_k = self.input_dim // self.n_layers
        print(x.shape)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        sub_q = einops.rearrange(q,
            'b n_patch (n_layers dim_k) -> b n_layers n_patch dim_k',
            n_layers=self.n_layers
        )
        sub_k = einops.rearrange(k,
            'b n_patch (n_layers dim_k) -> b n_layers n_patch dim_k',
            n_layers=self.n_layers
        )
        sub_v = einops.rearrange(v,
            'b n_patch (n_layers dim_k) -> b n_layers n_patch dim_k',
            n_layers=self.n_layers
        )
        attn_scores = einsum(sub_q, sub_k, 'b n i k, b n j k -> b n i j') / math.sqrt(dim_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        MHA = einsum(attn_weights, sub_v, 'b n i j, b n j d -> b n i d')
        H = einops.rearrange(MHA, 'b n_layers n_patch dim_k -> b n_patch (n_layers dim_k)')
        return self.Wo(H)

class Transformer(nn.Module):
    def __init__(self,n_layers=8,input_dim=None):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.Att=MultiHeadAttention(n_layers=n_layers,input_dim=input_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        x=x+self.Att(x)
        mean=torch.mean(x)
        std=torch.std(x)
        epsilon=1e-5
        x=(x-mean)/torch.sqrt(std*std+epsilon)
        x=self.alpha*x+self.beta
        return x


