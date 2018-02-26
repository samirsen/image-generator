'''
main.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as f
import constants
from gan_model import GAN
import util
import numpy as np

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
        gan.generate(text_caption_dict[k], noise_vec)
        break

if __name__ == '__main__':
    main()
