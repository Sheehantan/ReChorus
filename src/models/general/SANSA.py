import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from torch.nn.utils.rnn import pad_sequence
from models.BaseModel import GeneralModel

class SANSA(GeneralModel):
    @staticmethod
    def parse_model_args(parser):
        """
        Add additional arguments specific to the Sansa model.
        """
        parser.add_argument('--hidden_dim', type=int, default=64,
                            help='The dimensionality of the hidden layer.')
        parser.add_argument('--sparsity', type=float, default=0.01,
                            help='The sparsity level for approximate matrix inversion.')
        parser.add_argument('--regularization', type=float, default=0.01,
                            help='Regularization coefficient for matrix factorization.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.hidden_dim = args.hidden_dim
        self.sparsity = args.sparsity
        self.regularization = args.regularization

        # Define model parameters
        self._define_params()

    def _define_params(self):
        """
        Define the encoder and decoder layers based on the Sansa architecture.
        """
        # Encoder and decoder matrices
        self.W = nn.Parameter(torch.randn(self.item_num, self.hidden_dim) * 0.01)
        self.Z = nn.Parameter(torch.randn(self.hidden_dim, self.item_num) * 0.01)

        # Diagonal scaling for sparse approximate inversion
        self.D_inv = nn.Parameter(torch.ones(self.hidden_dim) * 0.01)

    def forward(self, feed_dict: dict) -> dict:
        """
        Perform a forward pass to compute predictions:
            r = (uW)Z
        """
        user_ids = feed_dict['user_id']
        item_ids = feed_dict['item_id']

        # One-hot encode user interactions
        user_interactions = torch.zeros((len(user_ids), self.item_num), device=self.device)
        user_interactions[torch.arange(len(user_ids), device=self.device).long(), user_ids.long()] = 1

        # Encoder layer: uW
        encoded_users = torch.matmul(user_interactions, self.W)

        # Perform diagonal scaling
        scaled_users = encoded_users / self.D_inv

        # Decoder layer: (uW)Z
        predictions = torch.matmul(scaled_users, self.Z[:, item_ids])

        return {'prediction': predictions}

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        Use a BPR loss for ranking.
        """
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()

        # Add regularization terms for W, Z, and D_inv
        reg_loss = self.regularization * (
            torch.norm(self.W, p=2) ** 2 + torch.norm(self.Z, p=2) ** 2 + torch.norm(self.D_inv, p=2) ** 2
        )

        return loss + reg_loss

    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index):
            """
            Construct the feed dictionary for a single instance.
            """
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }
            return feed_dict

        def actions_before_epoch(self):
            """
            Sample negative items for all instances before each training epoch.
            """
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
            for i, u in enumerate(self.data['user_id']):
                clicked_set = self.corpus.train_clicked_set[u]  # Exclude items already clicked
                for j in range(self.model.num_neg):
                    while neg_items[i][j] in clicked_set:
                        neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
            self.data['neg_items'] = neg_items