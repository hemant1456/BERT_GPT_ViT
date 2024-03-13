import torch.nn as nn
import math
import torch


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
    def forward(self, x):
        return self.embed(x) * self.scale


class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_embed = torch.zeros((seq_len,d_model))
        mul_factor = torch.exp(-(torch.arange(0,d_model,2))/d_model * math.log(10000)).unsqueeze(0) # exp(log(x)) is more stable than just x
        seq_embed = torch.arange(0,seq_len,1).unsqueeze(1)
        pos_embed[:, 0:d_model:2] = torch.sin(seq_embed* mul_factor)
        pos_embed[:, 1:d_model:2] = torch.cos(seq_embed* mul_factor)
        #to make it parameter with no gradient calculation we register it as parameter
        self.register_buffer('pe',pos_embed.unsqueeze(0)) #unsqueeze to handle batch size
    def forward(self,x):
        return self.dropout(x + self.pe[:,:x.size(1),:])