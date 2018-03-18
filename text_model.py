import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import constants
import collections
import functools
from util import *
from glove import Glove


class TextModel(nn.Module):
	def __init__(self, options):
		super(TextModel, self).__init__()

		self.options = options
		self.glove = Glove()
		self._GloVe = self.glove.get_embeddings()
		self.embeddings = nn.EmbeddingBag(constants.VOCAB_SIZE, constants.EMBED_DIM, mode=constants.REDUCE_TYPE)
		self.embeddings.weight = nn.Parameter(torch.tensor(self._GloVe))


	def forward(self, batch_input):
		"""
		Pass a batch of image captions through LSTM layer to learn embeddings.
		Input: python list of list ( batch_size x len(sentence_i) -- caption_word to glove index in main.py, content are indices in glove )
		Output: batch of average or sum of hidden representations from LSTM
		"""
		# batch_ftrs dim = batch_size x m x 300 -- issue might be torch.cat sentence has diff amount of words.
		# flatten, offsets -- flatten and keep track of index of starting word of each example
		# list of list and keep track of start word of each sentence
		flattened, offsets = preprocess(batch_input)
		embed_vecs = self.embeddings(flattened, offsets)
		return Variable(embed_vecs)  # These are (batch_size x 300) dim from mean of caption word embeddings.

	def backward(self):
		"""We use generator loss to update embeddings."""
		pass


class LSTM_Model(nn.Module):
	def __init__(self, options):
		super(LSTM_Model, self).__init__()

		self.options = options
		self.glove = Glove()
		self._GloVe = self.glove.get_embeddings()
		self.embeddings = nn.Embedding(num_embeddings=constants.VOCAB_SIZE, embedding_dim=constants.EMBED_DIM)
		self.embeddings.weight = nn.Parameter(self._GloVe) # Should this be here?
		self.embedding.weight.requires_grad = False   # Should this be here?

		self.biRNN = nn.LSTM(input_size=constants.EMBED_DIM, hidden_size=constants.HIDDEN_DIM,
						num_layers=1, batch_first=True, bidirectional=False)

		self.hidden = self.init_hidden(minibatch_size=constants.BATCH_SIZE)

	def init_hidden(self, minibatch_size):
		# Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (Variable(nn.init.xavier_uniform(1, minibatch_size, constants.HIDDEN_DIM)),
                Variable(nn.init.xavier_uniform(1, minibatch_size, constants.HIDDEN_DIM)))


	def forward(self, batch_input):
		"""
		Pass a batch of image captions through LSTM layer to learn embeddings.
		Input: python list of list ( batch_size x len(sentence_i) -- caption_word to glove index in main.py, content are indices in glove )
		Output: batch of average or sum of hidden representations from LSTM

		Requires sequence to be padded.
		"""
		_input = Variable(torch.LongTensor(batch_input))   # Potentially wrap as a Variable
		embed_vecs = self.embeddings(_input)

		# Now pass through LSTM cell => average hidden state => Generator

		# embed_vecs looks something like (batch_size, time_steps, 300 dim) ... each word is a time-step
		# (0 ,.,.) =
		#  -1.0822  1.2522  0.2434
		#   0.8393 -0.6062 -0.3348
		#   0.6597  0.0350  0.0837
		#   0.5521  0.9447  0.0498
		#
		# (1 ,.,.) =
		#   0.6597  0.0350  0.0837
		#  -0.1527  0.0877  0.4260
		#   0.8393 -0.6062 -0.3348
		#  -0.8738 -0.9054  0.4281
		input_seq = Variable(embed_vecs)  # maybe wrap as a Variable?
		lstm_out, self.hidden = self.biRNN(embed_vecs, self.hidden)

		# lstm_out dim = batch_size x seq_len x 300
		# return dim = batch_size x 300 -- governed by constants.REDUCE_TYPE
		generator_input = torch.mean(lstm_out, dim=1, keepdims=True)
		return Variable(generator_input)


	def backward(self):
		"""We use generator loss to update LSTM embedding weights."""
		pass
