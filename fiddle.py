import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
import constants
from model import Generator, Discriminator, BeganGenerator, BeganDiscriminator
from lstm_model import LSTM_Model
from vocab import get_glove
from util import *
from captions_utils import *
from train_utils import *
from data_batcher import *
import numpy as np
import matplotlib.pyplot as plt

def load_glove(paths):
    embeddings, word2id, id2word = (torch.load(path) for path in paths)
    return embeddings, word2id, id2word

def main():
    print("Starting to figure out GloVE embedding layer for lstm models ...")

    model_options = constants.MAIN_MODEL_OPTIONS
    caption_dict = load_flowers_capt_dict(data_dir='Data')   # filename --> [captions]
    img_dict = load_image_dict()   # filename --> 28 x 28 image

    if os.path.exists('Data/vocab/glove_matrix.torch'):
        paths = ['Data/vocab/glove_matrix.torch', 'Data/vocab/word_to_idx.torch', 'Data/vocab/idx_to_word.torch']
        embeddings, word2id, id2word = load_glove(paths)
    else:
        emb_matrix, word2id, id2word = get_glove(constants.GLOVE_PATH, constants.EMBED_DIM)
        embeddings = torch.from_numpy(emb_matrix).float()
        torch.save(embeddings, 'Data/vocab/glove_matrix.torch')
        torch.save(word2id, 'Data/vocab/word_to_idx.torch')
        torch.save(id2word, 'Data/vocab/idx_to_word.torch')

    print ( "shape of embedding size: ", embeddings.size() )


    # print len(word2id)
    # print len(id2word)
    #
    # print len(caption_dict)
    # print len(img_dict)
    #
    # print (embeddings[5, :])

    generator, discriminator = choose_model(model_options)
    g_optimizer, d_optimizer = choose_optimizer(generator, discriminator)

    ################################
    # Now get batch of captions and glove embeddings
    # Use this batch as input to BiRNN w LSTM cells
    ################################
    st = time.time()
    for i, batch_iter in enumerate(grouper(caption_dict.keys(), constants.BATCH_SIZE)):
        batch_keys = [x for x in batch_iter if x is not None]
        noise_vec = torch.randn(len(batch_keys), model_options['z_dim'], 1, 1)

        init_model(discriminator, generator)

        captions_batch, masks = get_captions_batch(batch_keys, caption_dict, word2id)
        





if __name__ == '__main__':
    main()
