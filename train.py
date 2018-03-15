'''
End-to-end training model
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import constants
from model import Generator, Discriminator, BeganGenerator, BeganDiscriminator
from text_model import TextModel, LSTM_Model
from util import *
from captions_utils import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip_longest
import scipy.misc
import matplotlib.pyplot as plt
import argparse
import time
import os

# From https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
# Iterates over an array in chunks
def grouper(array, n):
    args = [iter(array)] * n
    return izip_longest(*args)

# Show the generated image improves over time
def print_images(generated):
    for img in generated:
        image_done = img.data.numpy()
        swap_image = np.swapaxes(image_done,1,2)
        swap_image = np.swapaxes(swap_image,2,3)
        plt.imshow(swap_image[0])
        plt.show()

def get_text_description(text_caption_dict, batch_keys):
    g_idx = [np.random.randint(len(text_caption_dict[batch_keys[0]])) for i in range(len(batch_keys))]
    g_text_des = np.array([text_caption_dict[k][i] for k,i in zip(batch_keys, g_idx)])
    # g_text_des = np.expand_dims(g_text_des, axis=0) ONLY NEED FOR 1 DIM

    return g_text_des

def choose_wrong_image(image_dict, batch_keys):
    wrong_image = []
    for k in batch_keys:
        wrong_key = np.random.choice(image_dict.keys())
        while wrong_key == k:
            wrong_key = np.random.choice(image_dict.keys())

        wrong_image.append(image_dict[wrong_key])
    wrong_image = np.array(wrong_image)
    wrong_image = augment_image_batch(wrong_image)
    wrong_image = np.swapaxes(wrong_image, 2, 3)
    wrong_image = np.swapaxes(wrong_image, 1, 2)
    return wrong_image

# Finds the true image for the given batch data
def choose_true_image(image_dict, batch_keys):
    true_img = np.array([image_dict[k] for k in batch_keys])
    true_img = augment_image_batch(true_img)
    true_img = np.swapaxes(true_img, 2, 3)
    true_img = np.swapaxes(true_img, 1, 2)
    return true_img

def augment_image_batch(images):
    batch_size = images.shape[0]
    for i in range(batch_size):
        curr = images[i, :, :, :]
        if np.random.rand() > .5:
            curr = np.flip(curr, 1)
        images[i, :, :, :] = curr
    return images

def generate_step(text_caption_dict, noise_vec, batch_keys, generator):
    g_text_des = get_text_description(text_caption_dict, batch_keys)
    g_text_des = Variable(torch.Tensor(g_text_des))
    if torch.cuda.is_available():
        g_text_des = g_text_des.cuda()
    gen_image = generator.forward(g_text_des, noise_vec)   # Returns tensor variable holding image

    return gen_image

def make_save_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Made output directory")
    else:
        print("WARNING: starting training with an existing outputs directory")
    if not os.path.exists(output_path + 'weights/'):
        os.makedirs(output_path + 'weights/')
        print("Made weights directory")
    if not os.path.exists(output_path + 'images/'):
        os.makedirs(output_path + 'images/')
        print("Made images directory")


def choose_model(model_options):
    generator = Generator(model_options)
    discriminator = Discriminator(model_options)

    if torch.cuda.is_available():
        print("CUDA is available")
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        print("Moved models to GPU")

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    return generator, discriminator

def choose_optimizer(generator, discriminator):
    g_optimizer = optim.Adam(generator.parameters(), lr=constants.LR, betas=constants.BETAS)
    # Changes the optimizer to SGD if declared in constants
    if constants.D_OPTIMIZER_SGD:
        d_optimizer = optim.SGD(discriminator.parameters(), lr=constants.LR)
    else:
        d_optimizer = optim.Adam(discriminator.parameters(), lr=constants.LR, betas=constants.BETAS)

    print("Added optimizers")

    return g_optimizer, d_optimizer

def init_model(discriminator, generator):
    discriminator.train()
    generator.train()
    # Zero out gradient
    discriminator.zero_grad()
    for p in discriminator.parameters():
        p.requires_grad = True

def get_batches(caption_dict, img_dict, batch_keys, noise_vec):
    if torch.cuda.is_available():
        g_captions = torch.Tensor(get_text_description(caption_dict, batch_keys)).cuda()
        real_captions = torch.Tensor(get_text_description(caption_dict, batch_keys)).cuda()
        real_img = torch.Tensor(choose_real_image(img_dict, batch_keys)).cuda()
        wrong_img = torch.Tensor(choose_wrong_image(img_dict, batch_keys)).cuda()
        noise_vec = noise_vec.cuda()
    else:
        g_captions = torch.Tensor(get_text_description(caption_dict, batch_keys))
        real_captions = torch.Tensor(get_text_description(caption_dict, batch_keys))
        real_img = torch.Tensor(choose_real_image(img_dict, batch_keys))
        wrong_img = torch.Tensor(choose_wrong_image(img_dict, batch_keys))

    return g_captions, real_captions, real_img, wrong_img, noise_vec

def main():
    print("Starting training with LSTM ...")
    output_path = constants.SAVE_PATH
    make_save_dir(output_path)

    model_options = constants.MAIN_MODEL_OPTIONS
    caption_dict = load_flowers_capt_dict(data_dir='Data')
    img_dict = load_image_dict()

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
            print (gen_caption_batch.size())
            print (real_caption_batch.size())
            break

            # LSTM stuff goes here

            # gen_image = generator.forward(Variable(gen_captions), Variable(noise_vec))

if __name__ == '__main__':
    main()
