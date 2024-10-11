# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

    def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()

        # perform linear operation and split into h heads
        if not self.kq_same:
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
        output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(origin_shape)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None):
        """
        This is called by Multi-head attention object to find the values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = (scores - scores.max()).softmax(dim=-1)
        scores = scores.masked_fill(torch.isnan(scores), 0)
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
        super().__init__()
        """
        This is a Basic Block of Transformer. It contains one Multi-head attention object. 
        Followed by layer norm and position wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        context = self.masked_attn_head(seq, seq, seq, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output
    
        
class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x.clone()
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
