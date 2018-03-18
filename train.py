'''
End-to-end training model
'''
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
import numpy as np
import matplotlib.pyplot as plt

# From https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
# Iterates over an array in chunks
def main():
    print("Starting training with LSTM ...")
    output_path = constants.SAVE_PATH
    make_save_dir(output_path)

    model_options = constants.MAIN_MODEL_OPTIONS
    caption_dict = load_flowers_capt_dict(data_dir='Data')   # filename --> [captions]
    img_dict = load_image_dict()   # filename --> 28 x 28 image

    emb_matrix, word2id, id2word = get_glove(constants.GLOVE_PATH, constants.EMBED_DIM) 

    # lstm_model, lstm_optim = init_txt_model(model_options)
    generator, discriminator = choose_model(model_options)
    g_optimizer, d_optimizer = choose_optimizer(generator, discriminator)

    # Loop over dataset N times
    for epoch in range(constants.NUM_EPOCHS):
        print("Epoch %d" % (epoch))
        st = time.time()
        for i, batch_iter in enumerate(grouper(caption_dict.keys(), constants.BATCH_SIZE)):
            batch_keys = [x for x in batch_iter if x is not None]
            noise_vec = torch.randn(len(batch_keys), model_options['z_dim'], 1, 1)

            init_model(discriminator, generator)

            gen_caption_batch, real_caption_batch, real_img_batch, wrong_img_batch, noise_vec = get_batches(caption_dict, img_dict, batch_keys, noise_vec)

            # LSTM stuff goes here -- should input both gen_caption_batch and real_caption_batch to the lstm
            gen_embed_vecs = text_model.forward(gen_caption_batch)
            real_embed_vecs = text_model.forward(real_caption_batch)

            print gen_embed_vecs

            break
        break

            # gen_image = generator.forward(Variable(gen_captions), Variable(noise_vec))

if __name__ == '__main__':
    main()
