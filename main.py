'''
main.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as f
import constants
from gan_model import GAN


def main():
    model_options = constants.MAIN_MODEL_OPTIONS

    text_embed = [1,2,3,4,5]
    gan = GAN(model_options)
    gan.build_model(text_embed)

if __name__ == '__main__':
    main()
