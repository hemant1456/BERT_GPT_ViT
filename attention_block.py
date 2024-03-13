import torch.nn as nn
import math

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff: int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout= nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self,x):
        return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    '''
    class to implement multihead attention
    inputs: d_model: dimension of input, h: number of heads, dropout: %dropout
    '''
    def __init__(self, d_model: int, h:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout= nn.Dropout(dropout)
        assert d_model%h==0, "d_model is not divisible by h"
        self.d_k = d_model//h

        self.linear = nn.Linear(d_model, 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model)
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = query @ key.transpose(-2,-1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    def forward(self,x,mask):
        kqv = self.linear(x)
        query, key, value = kqv[:,:,:self.d_model], kqv[:,:,self.d_model: 2 * self.d_model], kqv[:,:,2 * self.d_model: 3 * self.d_model]
        query = query.view(query.shape[0], query.shape[1],self.h,  self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x ,self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous() #interchange head and seq_len dimension
        x = x.view(x.shape[0], x.shape[1], self.h * self.d_k)
        return self.w_o(x)