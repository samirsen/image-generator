'''
main.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as f
import constants
from gan_model import GAN
import util

def main():
    # Load the caption text vectors
    text_caption_dict = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)

    model_options = constants.MAIN_MODEL_OPTIONS
    gan = GAN(model_options)
    # gan.build_model(text_embed)

if __name__ == '__main__':
    main()
