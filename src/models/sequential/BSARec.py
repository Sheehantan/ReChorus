# -*- coding: UTF-8 -*-

# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BSARec
Reference:
    "BSARec: Enhanced Sequential Recommendation with Frequency-based and Self-attentive Models"
Note:
    Combining FrequencyLayer and MultiHeadAttention.
CMD example:
    python main.py --model_name BSARec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
    python main.py --model_name BSARec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""
import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers

class BSARec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'alpha', 'hidden_dropout_prob', 'hidden_size','c']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=2, help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads.')
        parser.add_argument('--alpha', type=float, default=0.9, help='Weighting parameter for DSP and GSP.')
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.5, help='Dropout probability for hidden layers.')
        parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layers.')
        parser.add_argument("--c", default=3, type=int)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.alpha = args.alpha
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.c = args.c
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.p_embeddings = nn.Embedding(self.max_his+1 , self.emb_size)
        self.frequency_layer = nn.ModuleList([
            layers.FrequencyLayer(self) for _ in range(self.num_layers)
        ])
        self.attention_layer = nn.ModuleList([
            layers.MultiHeadAttention(d_model=self.emb_size, n_heads=self.num_heads) for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']
        history = feed_dict['history_items']
        lengths = feed_dict['lengths'].long()
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        # Position embedding
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # Frequency Layer + Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        for i in range(self.num_layers):
            freq_output = self.frequency_layer[i](his_vectors)
            attn_output = self.attention_layer[i](his_vectors, his_vectors, his_vectors, attn_mask)
            his_vectors = self.alpha * freq_output + (1 - self.alpha) * attn_output

        his_vectors = his_vectors * valid_his[:, :, None].float()
        his_vector = his_vectors[torch.arange(batch_size).long(), (lengths - 1).long(), :]

        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(batch_size, -1)}
