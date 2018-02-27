'''
main.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import constants
from model import GAN
import util
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from itertools import izip_longest
import scipy.misc

np.random.seed(42)

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
    wrong_image = np.swapaxes(wrong_image, 2, 3)
    wrong_image = np.swapaxes(wrong_image, 1, 2)

    return wrong_image

def discrimate_step(gen_image, text_caption_dict, image_dict, batch_keys, gan):
    true_img = np.array([image_dict[k] for k in batch_keys])
    true_img = np.swapaxes(true_img, 2, 3)
    true_img = np.swapaxes(true_img, 1, 2)
    true_caption = get_text_description(text_caption_dict, batch_keys)

    wrong_img = choose_wrong_image(image_dict, batch_keys)

    real_img_passed = gan.discriminate(Variable(torch.Tensor(true_img)), Variable(torch.Tensor(true_caption)))
    wrong_img_passed = gan.discriminate(Variable(torch.Tensor(wrong_img)), Variable(torch.Tensor(true_caption)))
    fake_img_passed = gan.discriminate(gen_image, Variable(torch.Tensor(true_caption)))

    return real_img_passed, wrong_img_passed, fake_img_passed


def generate_step(text_caption_dict, noise_vec, batch_keys, gan):
    g_text_des = get_text_description(text_caption_dict, batch_keys)
    gen_image = gan.generate(g_text_des, noise_vec)   # Returns tensor variable holding image

    return gen_image


def main():
    model_options = constants.MAIN_MODEL_OPTIONS

    # Load the caption text vectors
    text_caption_dict = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)

    image_dict = util.load_images('Data/' + constants.DIRECTORY_PATH, text_caption_dict.keys())
    noise_vec = np.random.randn(constants.BATCH_SIZE, model_options['z_dim'])

    gan = GAN(model_options)

    # g_optimizer = optim.Adam(gan.parameters(), lr=0.0002, betas=(0.5, 0.25))
    # d_optimizer = optim.Adam(gan.parameters(), lr=0.0002, betas=(0.5, 0.25))

    # TODO: break text captions into multidimensional list
    # TODO: MAKE SURE IMAGES ARE OF DIMENSIONS (BATCHSIZE, CHANNELS, H, W)
    g_optimizer = optim.Adam(gan.g_model.parameters(), lr=0.0002, betas=(0.5, 0.25))
    d_optimizer = optim.Adam(gan.d_model.parameters(), lr=0.0002, betas=(0.5, 0.25))


    # Loop over dataset N times
    for epoch in range(constants.NUM_EPOCHS):
        print "Epoch: ", epoch
        for batch_iter in grouper(text_caption_dict.keys(), constants.BATCH_SIZE):
            batch_keys = [x for x in batch_iter if x is not None]
            # (BATCH, CHANNELS, H, W)  -- vectorized
            # (1, CHANNELS, H, W)
            gen_image = generate_step(text_caption_dict, noise_vec, batch_keys, gan)
            real_img_passed, wrong_img_passed, fake_img_passed = discrimate_step(gen_image, text_caption_dict, image_dict, batch_keys, gan)

            g_loss = gan.generator_loss(fake_img_passed)
            d_loss = gan.discriminator_loss(real_img_passed, wrong_img_passed, fake_img_passed)

            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        print 'G Loss: ', g_loss.data[0]
        print 'D Loss: ', d_loss.data[0]

        # Save images
        currImage = gen_image[0].data.numpy()
        currImage = np.swapaxes(currImage, 0, 1)
        currImage = np.swapaxes(currImage, 1,2)
        scipy.misc.imsave('Data/images/epoch' + str(epoch) + '.png', currImage)
        if epoch % 10 == 0:
            torch.save(gan.state_doct(), constants.SAVE_PATH + 'epoch' + str(epoch))



    # TESTING Discriminator
    # PYTORCH HAS DIMENSIONS (BATCHSIZE, CHANNELS, H, W)
    # NEED TO SWITCH FROM (BATCHSIZE, H, W, CHANNELS)
    # for i in image_dict:
    #     image_dict[i] = np.swapaxes(image_dict[i],1,2)
    #     image_dict[i] = np.swapaxes(image_dict[i],0,1)
    #     image_dict[i] = np.expand_dims(image_dict[i], axis=0)
    #     text_des = text_caption_dict[i][0]
    #     text_des = np.expand_dims(text_des, 0)
    #     output = gan.discriminate(Variable(torch.Tensor(image_dict[i])), Variable(torch.Tensor(text_des)))

    # print output

if __name__ == '__main__':
    main()
