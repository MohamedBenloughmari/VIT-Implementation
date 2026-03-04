import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from einops import rearrange

class AutoTokenizer(nn.Module):
    def __init__(self, patch_size=14, img_size=224, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(self.patch_dim, self.patch_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, self.patch_dim))

    def tokenize(self, img):
        patches = rearrange(
            img,
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1=self.patch_size,
            p2=self.patch_size
        )
        patches = self.projection(patches)
        return patches + self.pos_embedding



