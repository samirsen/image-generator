"""Batch the glove embeddings for the generator"""
from __future__ import absolute_import
from __future__ import division

import random
import time
import re

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID

import torch


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids


def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    masks = map(lambda x: [1] * len(x) + [0] * (maxlen - len(x)), token_batch)
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch), masks

def get_text_description(caption_dict, batch_keys):
    g_idx = [np.random.randint(len(caption_dict[batch_keys[0]])) for i in range(len(batch_keys))]
    g_text_des = [caption_dict[k][i] for k,i in zip(batch_keys, g_idx)]

    return g_text_des

def get_captions_batch(batch_keys, caption_dict, word2id):
    """
    Inputs:
        caption_dict: filename --> caption (dictionary)
        batch_keys: filenames in the batch
    Returns:
        batch of indices representing each sentence
    """
    tokens_batch = []
    raw_batch = get_text_description(caption_dict, batch_keys)
    for capt in raw_batch:
        tokens, ids = sentence_to_token_ids(capt, word2id)
        tokens_batch.append(ids)

    captions_batch, masks = padded(tokens_batch)
    return captions_batch, masks
