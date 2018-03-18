"""
Adding to code to start word embedding training.
Simple model to get the GloVe representations of
captions and take average of words for a
300 dimensional representation of caption.
"""

import torch
import torchtext.vocab as vocab
import numpy as np
import constants

class Glove:
    def __init__(self):
        # Gather GloVe 300 dimensional word embeddings
        self._glove = vocab.GloVe(name='6B', dim=constants.EMBED_DIM)

    # GloVe object includes attributes:
    # stoi (str-to-index) - returns dictionary of words to indexes
    # itos (index-to-str) returns an array of words by index
    def get_embeddings(self):
        return self._glove

    def get_index(self, word):
        return self._glove.stoi[word] 

    def get_word(self, word):
        """Returns vector representation for word"""
        return self._glove.vectors[self._glove.stoi[word]]

    def get_words(self, caption):
        """Returns numpy array of glove representations"""
        words = caption.lower().split()
        result = np.array([get_word[w] for w in words])
        return torch.Tensor(result)

    def closest_word(self, vector, n=10):
        """Find the most likely word for a given word vector"""
        all_dists = [(w, torch.dist(vector, get_word(w))) for w in glove.itos]
        return sorted(all_dists, key=lambda dist: dist[1][:n])

    def get_word_vectors(self, captions):
		# captions = Batch_size x 1
		batch_size = captions.shape[0]
		word_vecs = torch.Tensor()

		for i, caption in enumerate(captions):
			caption_rep = get_words(caption)
			word_vecs = torch.cat(word_vecs, caption_rep)

		return Variable(word_vecs)

    def _reduce_along_axis(self, word_vecs):
    	if constants.REDUCE_TYPE == "average":
    		word_vecs = torch.mean(word_vecs, dim=1)
    	elif constants.REDUCE_TYPE == "sum":
    		word_vecs = torch.sum(word_vecs, dim=1)

    	return word_vecs
