'''
main.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as f
import constants
from model import GAN
import util
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

np.random.seed(42)

def main():
    model_options = constants.MAIN_MODEL_OPTIONS


    # Load the caption text vectors
    text_caption_dict = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)
    image_dict = util.load_images('Data/' + constants.DIRECTORY_PATH, text_caption_dict.keys())
    # noise_vec = np.random.uniform(-1, 1, [len(text_caption_dict), model_options['z_dim']])
    noise_vec = np.random.randn(5, model_options['z_dim'])

    gan = GAN(model_options)

    # TODO: break text captions into multidimensional list
    # TODO: MAKE SURE IMAGES ARE OF DIMENSIONS (BATCHSIZE, CHANNELS, H, W)

    # TESTING GENERATOR

    generated = []               # Store images generated in each iteration
    for k in text_caption_dict:
        # image_done = gan.generate(text_caption_dict[k], noise_vec).data.numpy()
        idx = np.random.randint(len(text_caption_dict[k]))
        text_descript = text_caption_dict[k][idx]
        image_done = gan.generate(text_des, noise_vec)   # Returns tensor holding image
        generated.append(image_done)

        passed = gan.discriminate(Variable(image_done), Variable(torch.Tensor(np.expand_dims(text_descript, 0))))

        print passed

    # TESTING Discriminator
    # PYTORCH HAS DIMENSIONS (BATCHSIZE, CHANNELS, H, W)
    # NEED TO SWITHC TO (BATCHSIZE, H, W, CHANNELS)
    # for i in image_dict:
    #     image_dict[i] = np.swapaxes(image_dict[i],1,2)
    #     image_dict[i] = np.swapaxes(image_dict[i],0,1)
    #     image_dict[i] = np.expand_dims(image_dict[i], axis=0)
    #     text_des = text_caption_dict[i][0]
    #     text_des = np.expand_dims(text_des, 0)
    #     output = gan.discriminate(Variable(torch.Tensor(image_dict[i])), Variable(torch.Tensor(text_des)))

    # print output
    # # Swap axes of the image

    # Show the generated image improves over time 
    def print_images(generated):
        for img in generated:
            swap_image = np.swapaxes(image_done,1,2)
            swap_image = np.swapaxes(swap_image,2,3)
            plt.imshow(swap_image[0])
            plt.show()

if __name__ == '__main__':
    main()
