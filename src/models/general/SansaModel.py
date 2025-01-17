import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import spsolve, LinearOperator
from typing import Dict, List
from models.BaseModel import GeneralModel
class SansaModel(GeneralModel):
	reader, runner = 'BaseReader', 'BaseRunner'
	
	@staticmethod
	def parse_model_args(parser):
		parser = GeneralModel.parse_model_args(parser)
		parser.add_argument('--embedding_dim', type=int, default=64,
							help='Dimension of the embedding.')
		parser.add_argument('--lambda_reg', type=float, default=0.1,
							help='Regularization parameter for the matrix inversion.')
		return parser
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.embedding_dim = args.embedding_dim
		self.lambda_reg = args.lambda_reg
		
		self._define_params()
		self.apply(self.init_weights)
		
		# Precompute sparse matrices if possible to speed up training.
		self.precomputed_W_T, self.precomputed_Z = self._precompute_sparse_matrices(corpus)

	def _define_params(self):
		# Define the parameters for the model here.
		self.user_embeddings = nn.Embedding(self.user_num, self.embedding_dim)
		self.item_embeddings = nn.Embedding(self.item_num, self.embedding_dim)

		# We will not define additional parameters here because sansa uses a specific method to calculate embeddings.

	def _precompute_sparse_matrices(self, corpus):
		"""
		Precompute sparse matrices for faster training.
		This is where you would perform the Cholesky decomposition and find the sparse inverse.
		"""
		# Construct the interaction matrix X (user-item interactions) from the corpus data.
		# Then compute A = X^T * X + lambda * I, where I is the identity matrix.
		# Finally, approximate the inverse of A using a suitable method.
		# This part is highly dependent on the specifics of your dataset and how it's represented in the corpus.
		# It may require converting user-item interactions into a sparse matrix format and then applying the algorithm described in the paper.

		# For demonstration purposes, we'll create a dummy interaction matrix.
		X = self._create_interaction_matrix(corpus)

		# Compute A = X^T * X + lambda * I
		A = X.T @ X + self.lambda_reg * identity(X.shape[1])

		# Perform Cholesky decomposition: A ≈ L D L^T
		L, D = self._sparse_cholesky_decomposition(A)

		# Approximate the inverse of L: K ≈ L^-1
		K = self._approximate_inverse(L)

		# Compute W and Z as described in the paper
		W_T = K.T
		Z0 = spsolve(diags(D), W_T)
		r = np.diag(W_T @ Z0)
		Z = Z0 * (-1 / r)

		return W_T, Z

	def _create_interaction_matrix(self, corpus):
		# Create a sparse matrix representation of user-item interactions.
		rows, cols, data = [], [], []
		for u, items in enumerate(corpus.train_clicked_set):
			for i in items:
				rows.append(u)
				cols.append(i)
				data.append(1)
		X = csr_matrix((data, (rows, cols)), shape=(self.user_num, self.item_num))
		return X

	def _sparse_cholesky_decomposition(self, A):
		# Placeholder for actual implementation of sparse Cholesky decomposition.
		# Use an appropriate library or method to perform this step.
		# For example, you can use scipy.sparse.linalg.splu or another method.
		# Here we assume A is already positive definite and symmetric.
		# Return L and D such that A ≈ L D L^T.
		from scipy.sparse.linalg import spilu  # Incomplete LU factorization as an approximation
		ilu = spilu(A.tocsc())
		L = ilu.L.tocsr()
		D = ilu.U.diagonal()
		return L, D

	def _approximate_inverse(self, L):
		# Placeholder for actual implementation of sparse approximate inverse.
		# Use an appropriate method to find a sparse approximate inverse of L.
		# Here we use a simple iterative method for demonstration.
		I = identity(L.shape[0], format='csr')
		K = 2 * I - L
		return K

	def forward(self, feed_dict: dict) -> dict:
		"""
		Define the forward propagation logic for the sansa model.
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		user_ids = feed_dict['user_id']
		item_ids = feed_dict['item_id']

		# Get user embeddings
		user_embs = self.user_embeddings(user_ids).cpu().detach().numpy()

		# Convert item_ids to one-hot encoding and apply W_T and Z
		item_embs = self.precomputed_W_T[item_ids] @ self.precomputed_Z

		# Convert back to tensor for final prediction
		prediction = torch.tensor(np.dot(user_embs, item_embs.T), device=self.device)

		out_dict = {'prediction': prediction}
		return out_dict

