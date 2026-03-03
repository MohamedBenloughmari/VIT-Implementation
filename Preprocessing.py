import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from einops import rearrange

class AutoTokenizer:
    def __init__(self, patch_size=14, embedding_dim=64):
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
    
    def tokenize(self, img):
        patches = rearrange(
            img,
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1=self.patch_size,
            p2=self.patch_size
        )
        return patches



