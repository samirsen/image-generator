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

np.random.seed(42)

# Show the generated image improves over time
def print_images(generated):
    for img in generated:
        image_done = img.data.numpy()
        swap_image = np.swapaxes(image_done,1,2)
        swap_image = np.swapaxes(swap_image,2,3)
        plt.imshow(swap_image[0])
        plt.show()

def get_text_description(text_caption_dict, k):
    g_idx = np.random.randint(len(text_caption_dict[k]))
    g_text_des = text_caption_dict[k][g_idx]
    g_text_des = np.expand_dims(g_text_des, axis=0)

    return g_text_des

def choose_wrong_image(image_dict, k):
    wrong_idx = np.random.randint(0, len(image_dict))
    while wrong_idx == k:
        wrong_idx = np.random.randint(0, len(image_dict))

    wrong_image = image_dict[wrong_idx]
    wrong_image = np.swapaxes(wrong_image, 1, 2)
    wrong_image = np.swapaxes(wrong_image, 2, 3)

    return wrong_image

def discrimate_step(gen_image, text_caption_dict, image_dict, k, gan):
    true_img = image_dict[k]
    true_img = np.swapaxes(true_img, 1, 2)
    true_img = np.swapaxes(true_img, 0, 1)
    true_caption = get_text_description(text_caption_dict, k)

    wrong_img = choose_wrong_image(image_dict, k)

    real_img_passed = gan.discriminate(Variable(torch.Tensor(true_img)), Variable(torch.Tensor(text_des)))
    wrong_img_passed = gan.discriminate(Variable(torch.Tensor(wrong_img)), Variable(torch.Tensor(text_des)))
    fake_img_passed = gan.discriminate(gen_image, Variable(torch.Tensor(text_des)))

    return real_img_passed, wrong_img_passed, fake_img_passed


def generate_step(text_caption_dict, noise_vec, k, gan):
    g_text_des = get_text_description(text_caption_dict, k)
    gen_image = gan.generate(g_text_des, noise_vec)   # Returns tensor variable holding image

    return gen_image

def main():
    model_options = constants.MAIN_MODEL_OPTIONS

    # Load the caption text vectors
    text_caption_dict = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)
    image_dict = util.load_images('Data/' + constants.DIRECTORY_PATH, text_caption_dict.keys())
    noise_vec = np.random.randn(constants.BATCH_SIZE, model_options['z_dim'])

    gan = GAN(model_options)
    g_optimizer = optim.Adam(gan.parameters(), lr=0.0002, betas=(0.5, 0.25))
    d_optimizer = optim.Adam(gan.parameters(), lr=0.0002, betas=(0.5, 0.25))
    # TODO: break text captions into multidimensional list
    # TODO: MAKE SURE IMAGES ARE OF DIMENSIONS (BATCHSIZE, CHANNELS, H, W)

    # Loop over dataset N times
    for epoch in range(1):

        generated = []
        for k in text_caption_dict:
            # (BATCH, CHANNELS, H, W)  -- vectorized
            # (1, CHANNELS, H, W)
            gen_image = generate_step(text_caption_dict, noise_vec, k, gan)
            real_img_passed, wrong_img_passed, fake_img_passed = discrimate_step(gen_image, text_caption_dict, image_dict, k, gan)

            #TODO Add loss and update
            g_loss = gan.generator_loss(fake_img_passed)
            d_loss = gan.discriminator_loss(real_img_passed, wrong_img_passed, fake_img_passed)

            g_loss.backward()
            g_optimzer.step()

            d_loss.backward()
            d_optimizer.step() 

            generated.append(gen_image)


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
