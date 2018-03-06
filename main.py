'''
main.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import constants
from model import Generator, Discriminator
import util
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip_longest
import scipy.misc
import matplotlib.pyplot as plt

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

def discrimate_step(gen_image, text_caption_dict, image_dict, batch_keys, discriminator):
    true_img = np.array([image_dict[k] for k in batch_keys])
    true_img = np.swapaxes(true_img, 2, 3)
    true_img = np.swapaxes(true_img, 1, 2)
    true_caption = get_text_description(text_caption_dict, batch_keys)

    wrong_img = choose_wrong_image(image_dict, batch_keys)

    real_img_passed = discriminator.forward(Variable(torch.Tensor(true_img)), Variable(torch.Tensor(true_caption)))
    wrong_img_passed = discriminator.forward(Variable(torch.Tensor(wrong_img)), Variable(torch.Tensor(true_caption)))
    fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)))

    return real_img_passed, wrong_img_passed, fake_img_passed


def generate_step(text_caption_dict, noise_vec, batch_keys, generator):
    g_text_des = get_text_description(text_caption_dict, batch_keys)
    g_text_des = Variable(torch.Tensor(g_text_des))
    gen_image = generator.forward(g_text_des, noise_vec)   # Returns tensor variable holding image

    return gen_image


def main():
    model_options = constants.MAIN_MODEL_OPTIONS

    # Load the caption text vectors
    text_caption_dict = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)

    image_dict = util.load_images('Data/' + constants.DIRECTORY_PATH, text_caption_dict.keys())
    noise_vec = Variable(torch.randn(constants.BATCH_SIZE, model_options['z_dim'], 1, 1))

    generator = Generator(model_options)
    discriminator = Discriminator(model_options)

    # Initialize weights
    generator.apply(util.weights_init)
    discriminator.apply(util.weights_init)

    g_optimizer = optim.Adam(generator.parameters(), lr=constants.LR, betas=constants.BETAS)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=constants.LR, betas=constants.BETAS)


    # TODO: Do we need to choose all of the images and captions before training or continuously choose new ones?

    # TODO: MAKE SURE IMAGES ARE OF DIMENSIONS (BATCHSIZE, CHANNELS, H, W)
    # TODO: ADD L1/L2 Regularizaiton
    # TODO: USE DATALOADER FROM TORCH UTILS!!!!!!!!!
    # TODO: OPTIMIZE FOR GPU (CUDA)
    # TODO: ADD PARALLELIZATION
    # TODO: ADD IMAGE PREPROCESSING? DO WE NEED TO SUBTRACT/ADD ANYTHING TO IMAGES

    # data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    # Loop over dataset N times
    for epoch in range(constants.NUM_EPOCHS):
        print ("Epoch: %d" %(epoch))
        for batch_iter in grouper(text_caption_dict.keys(), constants.BATCH_SIZE):
            batch_keys = [x for x in batch_iter if x is not None]

            # Zero out gradient
            generator.zero_grad()
            discriminator.zero_grad()

            # Run through generator and discriminator
            gen_image = generate_step(text_caption_dict, noise_vec, batch_keys, generator)
            real_img_passed, wrong_img_passed, fake_img_passed = discrimate_step(gen_image, text_caption_dict, image_dict, batch_keys, discriminator)

            # Calculate loss
            g_loss = generator.loss(fake_img_passed)
            d_loss = discriminator.loss(real_img_passed, wrong_img_passed, fake_img_passed)

            # Update optimizer
            g_loss.backward(retain_graph=True)
            g_optimizer.step()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        print 'G Loss: ', g_loss.data[0]
        print 'D Loss: ', d_loss.data[0]

        # Save images
        currImage = gen_image[0].data.numpy()
        currImage = np.swapaxes(currImage, 0, 1)
        currImage = np.swapaxes(currImage, 1, 2)
        scipy.misc.imsave('Data/images/epoch' + str(epoch) + '.png', currImage)
        # Save model
        if epoch % 10 == 0 or epoch == constants.NUM_EPOCHS - 1:
            torch.save(generator.state_dict(), constants.SAVE_PATH + 'g_epoch' + str(epoch))
            torch.save(discriminator.state_dict(), constants.SAVE_PATH + 'd_epoch' + str(epoch))



    # FOR TESTING
    # for k in text_caption_dict:
    #     noise_vec = torch.randn(5, model_options['z_dim'], 1, 1)
    #     image = generator.forward(Variable(torch.Tensor(text_caption_dict[k])), Variable(torch.Tensor(noise_vec)))
    #     output = discriminator.forward(image, Variable(torch.Tensor(text_caption_dict[k])))
    #     print "DISCRIM OUTPUT", output
    #     break
    # print image.shape
    # swap_image = image.data.numpy()[0]
    # swap_image = np.swapaxes(swap_image,0,1)
    # swap_image = np.swapaxes(swap_image,1,2)
    # print swap_image.shape
    # plt.imshow(swap_image)
    # plt.show()
    # END TESTING

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
