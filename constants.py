'''
constants.py
Holds all of the constants info of the project
'''
import torch.nn as nn

'''
data_loader.py constants
'''
# The entire 102 Category Flower Dataset
ENTIRE_DATASET = 'flowers/jpg/'
# A small subset (the first 128 flower pictures) of the entire flower dataset (used for testing code out)
SMALL_DATASET = 'flowers/smalljpg/'

# Choose which directory holds the images you want for the flowers
DIRECTORY_PATH = SMALL_DATASET

# The name of the output file that the word vectors will be saved as.
# This file name should end with '.hdf5'
VEC_OUTPUT_FILE_NAME = 'flower_tv.hdf5'



'''
Model Constants
main.py, gan_model.py
'''
# If true, prints status of creating model
PRINT_MODEL_STATUS = True

# Three channels for Red, Green, and Blue
RGB_CHANNELS = 3
# Number of pixels in one Dimension of an image
IMAGE_SIZE = 128
# The basis for the number of channels in the generator (everything else is a multiple of this)
G_CHANNELS = 64
# The basis for the number of channels in the discriminator (everything else is a multiple of this)
D_CHANNELS = 64
# The main padding for each layer in the GAN
MAIN_PADDING = 2
# The main stride for each layer in the GAN
MAIN_STRIDE = 2

# Expands the embedded text vectors in the discriminator
D_EMBED_EXPAND = 8

# The batch size for training
# BATCH_SIZE = 64
BATCH_SIZE = 2


MAIN_MODEL_OPTIONS = {
    'caption_vec_len':4800,     # Dimensions for the embedded captions vector
    't_dim':256,                # Dimensions for the text vector inputted into the GAN
    'z_dim':100,
    'batch_size':BATCH_SIZE,
    'image_size':IMAGE_SIZE,
    'g_channels':G_CHANNELS,          # The basis for the number of channels in the generator
    'd_channels':D_CHANNELS,
    'leak':0.2,                         # Leak for Leaky ReLU
    'bn_momentum':0.9,                  # Batch norm momentum
    'bn_eps':1e-05,                     # Batch norm epsilon
    # Generator constants
    'g_num_layers':4,                 # Number of layers in the generator
    'g_layer_conv_sizes':[            # List of shape of each layer (number entries, num channels, height, width), starting at input layer
                                [-1, G_CHANNELS * 8, int(IMAGE_SIZE/16), int(IMAGE_SIZE/16)],
                                [-1, G_CHANNELS * 4, int(IMAGE_SIZE/8), int(IMAGE_SIZE/8)],
                                [-1, G_CHANNELS * 2, int(IMAGE_SIZE/4), int(IMAGE_SIZE/4)],
                                [-1, G_CHANNELS * 1, int(IMAGE_SIZE/2), int(IMAGE_SIZE/2)],
                                [-1, RGB_CHANNELS, int(IMAGE_SIZE), int(IMAGE_SIZE)],
                            ],
    'g_layer_filter_sizes':[          # List of filter sizes for each layer of the GAN (starts with input, which we will ignore)
                                0,
                                5,
                                5,
                                5,
                                5
                                ],
    'g_layer_num_channels':[          # List of the number of channels for each layer of the GAN (starts with input)
                                G_CHANNELS*8,
                                G_CHANNELS*4,
                                G_CHANNELS*2,
                                G_CHANNELS,
                                RGB_CHANNELS
                                ],
    'g_layer_padding':[          # Padding for each layer of the GAN (starts with input)
                                None,
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING)
                                ],
    'g_layer_stride':[          # Stride for each layer of the GAN (starts with input)
                                None,
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE)
                                ],
    'g_layer_activation_func':[       # List of the activations functions to be applied to each layer (empty string is no activation function)
                                    '',
                                    'relu',
                                    'relu',
                                    'relu',
                                    'tanh'
                                    ],
    # Discriminator constants
    'd_num_layers':4,                   # Number of layers in the discriminator
    'd_layer_conv_sizes':[            # List of shape of each layer (number entries, num channels, height, width), starting at input layer
                                [-1, RGB_CHANNELS, int(IMAGE_SIZE), int(IMAGE_SIZE)],
                                [-1, D_CHANNELS * 1, int(IMAGE_SIZE/2), int(IMAGE_SIZE/2)],
                                [-1, D_CHANNELS * 2, int(IMAGE_SIZE/4), int(IMAGE_SIZE/4)],
                                [-1, D_CHANNELS * 4, int(IMAGE_SIZE/8), int(IMAGE_SIZE/8)],
                                [-1, D_CHANNELS * 8, int(IMAGE_SIZE/16), int(IMAGE_SIZE/16)],
                            ],
    'd_layer_filter_sizes':[          # List of filter sizes for each layer of the GAN (starts with input, which we will ignore)
                                0,
                                5,
                                5,
                                5,
                                5
                                ],
    'd_layer_num_channels':[          # List of the number of channels for each layer of the GAN (starts with input)
                                RGB_CHANNELS,
                                D_CHANNELS * 1,
                                D_CHANNELS * 2,
                                D_CHANNELS * 4,
                                D_CHANNELS * 8
                                ],
    'd_layer_padding':[          # Padding for each layer of the GAN (starts with input)
                                None,
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                ],
    'd_layer_stride':[          # Stride for each layer of the GAN (starts with input)
                                None,
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                ],
    'd_layer_activation_func':[       # List of the activations functions to be applied to each layer (empty string is no activation function)
                                    '',
                                    'lrelu',
                                    'lrelu',
                                    'lrelu',
                                    'lrelu'
                                    ],
    }
