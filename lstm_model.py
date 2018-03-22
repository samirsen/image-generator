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

class LSTM(nn.Module):
    def __init__(self, options, embeddings):
        print ("Initializing LSTM module ...")
        super(LSTM, self).__init__()

        self.options = options
        self.hidden_dim = constants.HIDDEN_DIM
        # self.encoder = nn.Embedding.from_pretrained(embeddings)
        self.encoder = nn.Embedding(constants.VOCAB_SIZE, constants.EMBED_DIM)
        self.encoder.weight = nn.Parameter(embeddings)

        self.hidden = self.init_hidden()
        self.biRNN = nn.LSTM(input_size=constants.EMBED_DIM, hidden_size=constants.HIDDEN_DIM,
						num_layers=constants.NUM_LAYERS, batch_first=True, bidirectional=constants.BIDIRECTIONAL)

    def init_hidden(self, batch=constants.BATCH_SIZE):
        """Initialize the hidden state for bidirectional/lstm model as random small numbers."""

        return (Variable(torch.randn(constants.NUM_LAYERS * constants.NUM_DIRECTIONS, batch, self.hidden_dim)),
                Variable(torch.randn(constants.NUM_LAYERS * constants.NUM_DIRECTIONS, batch, self.hidden_dim)))

    def average_glove_embeddings(self, caption_batch, masks):
         _input = Variable(torch.LongTensor(captions_batch))
         embeds = self.encoder(_input)

         corrected_masks = Variable(masks.view(captions_batch.shape[0], captions_batch.shape[1], 1))
         masked_out = torch.mul(corrected_masks, embeds)
         return torch.mean(masked_out, dim=1, keepdim=True)

    def forward(self, captions_batch, masks):
        """
        Inputs:
            numpy array of list of ints for the indices in embeddings matrix where sequences padded to maxlength sequence

        Pass glove embeddings for padded sequence of batch of captions into BiLSTM.
        Average the hidden states' vector representations as the learned value.
        """

        _input = Variable(torch.LongTensor(captions_batch))
        embeds = self.encoder(_input)

        lstm_in = embeds.cuda()
        lstm_out, self.hidden = self.biRNN(lstm_in, self.hidden)

        corrected_masks = Variable(masks.view(captions_batch.shape[0], captions_batch.shape[1], 1))
        masked_lstm_out = torch.mul(corrected_masks, lstm_out)

        averaged_hidden = torch.mean(masked_lstm_out, dim=1, keepdim=True)
        return averaged_hidden
