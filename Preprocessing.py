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
            'c (h p) (w p) -> (h w) (c p p)',
            p=self.patch_size
        )
        return patches



