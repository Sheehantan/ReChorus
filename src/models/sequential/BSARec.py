import os
import random
import torch
import numpy as np
import logging
import argparse

from models.BaseModel import SequentialModel
from utils import layers
import torch.nn as nn

class FrequencyLayer(nn.Module):
    def __init__(self, d_model, c, dropout=0):
        super().__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, input_tensor):
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        low_pass = x.clone()
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=input_tensor.size(1), dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BSARecLayer(nn.Module):
    def __init__(self, args, d_model, n_heads, c, alpha, dropout=0):
        super().__init__()
        self.filter_layer = FrequencyLayer(d_model, c, dropout=dropout)
        self.attention_layer = layers.MultiHeadAttention(args)  # 传递 args
        self.alpha = alpha

    def forward(self, input_tensor, attention_mask=None):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + (1 - self.alpha) * gsp
        return hidden_states

class BSARecBlock(nn.Module):
    def __init__(self, args, d_model, d_ff, n_heads, c, alpha, dropout=0):
        super().__init__()
        self.layer = BSARecLayer(args, d_model, n_heads, c, alpha, dropout=dropout)
        self.feed_forward = layers.FeedForward(d_model, d_ff, dropout=args.hidden_dropout_prob)  # 使用 args 中的 dropout 参数

    def forward(self, hidden_states, attention_mask=None):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output




class BSARec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--emb_size", default=64, type=int, help="Size of embedding vectors.")
        parser.add_argument("--num_layers", default=2, type=int, help="Number of BSARec layers.")
        parser.add_argument("--num_heads", default=4, type=int, help="Number of attention heads.")
        parser.add_argument("--c", default=3, type=int, help="Frequency component parameter.")
        parser.add_argument("--alpha", default=0.9, type=float, help="Weight for combining frequency and attention output.")
        parser.add_argument("--hidden_size", default=64, type=int, help="Hidden size of attention mechanism.")  # 添加hidden_size
        parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout probability for attention scores.")  # 添加attention_probs_dropout_prob
        parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout probability for hidden layers.")  # 添加hidden_dropout_prob
        
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.c = args.c
        self.alpha = args.alpha
        self.hidden_size = args.hidden_size
        self.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        self.hidden_dropout_prob = args.hidden_dropout_prob

        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            BSARecBlock(args, d_model=self.emb_size, d_ff=self.emb_size * 4, n_heads=self.num_heads,
                        c=self.c, alpha=self.alpha, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
        
        self.apply(self.init_weights)

    def forward(self, feed_dict):
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        # Position embedding
        position = (lengths[:, None] - torch.arange(seq_len).to(self.device)) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)

        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)

        his_vectors = his_vectors * valid_his[:, :, None].float()

        his_vector = his_vectors[torch.arange(batch_size).long(), (lengths - 1).long(), :]

        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(batch_size, -1)}


    def calculate_loss(self, feed_dict):
        seq_output = self.forward(feed_dict)
        seq_output = seq_output['prediction']
        pos_pred, neg_pred = seq_output[:, 0], seq_output[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
        return loss
