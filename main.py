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

def main():
    model_options = constants.MAIN_MODEL_OPTIONS

    # Load the caption text vectors
    text_caption_dict = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)
    image_dict = util.load_images('Data/' + constants.DIRECTORY_PATH, text_caption_dict.keys())
    # noise_vec = np.random.uniform(-1, 1, [len(text_caption_dict), model_options['z_dim']])
    noise_vec = np.random.randn(5, model_options['z_dim'])

    gan = GAN(model_options)

    # TODO: break text captions into multidimensional list
    for k in text_caption_dict:
        image_done = gan.generate(text_caption_dict[k], noise_vec).data.numpy()
        break

    # Swap axes of the image
    print "swapping"
    swap_image = np.swapaxes(image_done,1,2)
    swap_image = np.swapaxes(swap_image,2,3)
    print swap_image.shape
    plt.imshow(swap_image[0])
    plt.show()

if __name__ == '__main__':
    main()
