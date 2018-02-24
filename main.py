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
    # Load
    text_caption_dict = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)
    print text_caption_dict
    # gan = GAN(constants.MAIN_MODEL_OPTIONS)
    # gan.build_model(text_embed)

if __name__ == '__main__':
    main()
