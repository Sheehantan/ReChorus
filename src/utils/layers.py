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

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_heads, kq_same=False, bias=True):
#         super().__init__()
#         """
#         It has projection layer for getting keys, queries and values. Followed by attention.
#         """
#         self.d_model = d_model
#         self.h = n_heads
#         self.d_k = self.d_model // self.h
#         self.kq_same = kq_same

#         if not kq_same:
#             self.q_linear = nn.Linear(d_model, d_model, bias=bias)
#         self.k_linear = nn.Linear(d_model, d_model, bias=bias)
#         self.v_linear = nn.Linear(d_model, d_model, bias=bias)

#     def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
#         new_x_shape = x.size()[:-1] + (self.h, self.d_k)
#         return x.view(*new_x_shape).transpose(-2, -3)

#     def forward(self, q, k, v, mask=None):
#         origin_shape = q.size()

#         # perform linear operation and split into h heads
#         if not self.kq_same:
#             q = self.head_split(self.q_linear(q))
#         else:
#             q = self.head_split(self.k_linear(q))
#         k = self.head_split(self.k_linear(k))
#         v = self.head_split(self.v_linear(v))

#         # calculate attention using function we will define next
#         output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

#         # concatenate heads and put through final linear layer
#         output = output.transpose(-2, -3).reshape(origin_shape)
#         return output

#     @staticmethod
#     def scaled_dot_product_attention(q, k, v, d_k, mask=None):
#         """
#         This is called by Multi-head attention object to find the values.
#         """
#         scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -np.inf)
#         scores = (scores - scores.max()).softmax(dim=-1)
#         scores = scores.masked_fill(torch.isnan(scores), 0)
#         output = torch.matmul(scores, v)  # bs, head, q_len, d_k
#         return output

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        if args.hidden_size % args.num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_heads))
        self.args = args
        self.num_attention_heads = args.num_heads
        self.attention_head_size = int(args.hidden_size / args.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12) # TODO
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        #Add & Norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
#         super().__init__()
#         """
#         This is a Basic Block of Transformer. It contains one Multi-head attention object. 
#         Followed by layer norm and position wise feedforward net and dropout layer.
#         """
#         # Multi-Head Attention Block
#         self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

#         # Two layer norm layer and two dropout layer
#         self.layer_norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.linear2 = nn.Linear(d_ff, d_model)

#         self.layer_norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, seq, mask=None):
#         context = self.masked_attn_head(seq, seq, seq, mask)
#         context = self.layer_norm1(self.dropout1(context) + seq)
#         output = self.linear1(context).relu()
#         output = self.linear2(output)
#         output = self.layer_norm2(self.dropout2(output) + context)
#         return output
    
        
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
    
class BSARecLayer(nn.Module):
    def __init__(self, d_model, n_heads, c, alpha, dropout=0):
        super().__init__()
        self.filter_layer = FrequencyLayer(d_model, c, dropout=dropout)
        self.attention_layer = MultiHeadAttention(d_model, n_heads)
        self.alpha = alpha

    def forward(self, input_tensor, attention_mask=None):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, input_tensor, input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + (1 - self.alpha) * gsp
        return hidden_states

import math

def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(FeedForward, self).__init__()

        self.dense_1 = nn.Linear(d_model, d_ff)  # d_ff 通常是 d_model 的四倍
        self.intermediate_act_fn =gelu  # 使用 gelu 作为激活函数

        self.dense_2 = nn.Linear(d_ff, d_model)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)  # 使用 nn.LayerNorm 进行归一化
        self.dropout = nn.Dropout(dropout)  # dropout 概率

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)  # 使用 gelu 激活函数

        hidden_states = self.dense_2(hidden_states)  # 再次线性变换
        hidden_states = self.dropout(hidden_states)  # 应用 dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 残差连接和归一化

        return hidden_states
