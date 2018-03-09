"""
Adding to code to start word embedding training.
Simple model to get the GloVe representations of
captions and take average of words for a
300 dimensional representation of caption.
"""

import torch
import torchtext.vocab

# Gather GloVe 300 dimensional word embeddings
glove = vocab.GloVe(name='6B', dim=300)

# GloVe object includes attributes:
# stoi (str-to-index) - returns dictionary of words to indexes
# itos (index-to-str) returns an array of words by index

def get_word(word):
    """Returns vector representation for word"""
    return glove.vectors[glove.stoi[word]]

def closest_word(vector, n=10):
    """Find the most likely word for a given word vector"""
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in glove.itos]
    return sorted(all_dists, key=lambda dist: dist[1])[:n])
