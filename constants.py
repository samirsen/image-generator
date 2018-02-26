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
IMAGE_SIZE = 64
# The basis for the number of channels in the generator (everything else is a multiple of this)
GAN_CHANNELS = 64
# The main padding for each layer in the GAN
MAIN_PADDING = 2
# The main stride for each layer in the GAN
MAIN_STRIDE = 2

# The batch size for training
BATCH_SIZE = 64


MAIN_MODEL_OPTIONS = {
    'caption_vec_len':4800,     # Dimensions for the embedded captions vector
    't_dim':256,                # Dimensions for the text vector inputted into the GAN
    'leak':0.2,             # Leak for Leaky ReLU
    'z_dim':100,
    'batch_size':BATCH_SIZE,
    'image_size':IMAGE_SIZE,
    'g_channels':GAN_CHANNELS,          # The basis for the number of channels in the generator
    'd_channels':None,
    'gan_num_layers':4,                 # Number of layers in the GAN
    'gan_layer_conv_sizes':[            # List of shape of each layer (number entries, num channels, height, width), starting at input layer
                                [-1, GAN_CHANNELS * 8, int(IMAGE_SIZE/16), int(IMAGE_SIZE/16)],
                                [BATCH_SIZE, GAN_CHANNELS * 4, int(IMAGE_SIZE/8), int(IMAGE_SIZE/8)],
                                [BATCH_SIZE, GAN_CHANNELS * 2, int(IMAGE_SIZE/4), int(IMAGE_SIZE/4)],
                                [BATCH_SIZE, GAN_CHANNELS * 1, int(IMAGE_SIZE/2), int(IMAGE_SIZE/2)],
                                [BATCH_SIZE, RGB_CHANNELS, int(IMAGE_SIZE), int(IMAGE_SIZE)],
                            ],
    'gan_layer_filter_sizes':[          # List of filter sizes for each layer of the GAN (starts with input, which we will ignore)
                                0,
                                5,
                                5,
                                5,
                                5
                                ],
    'gan_layer_num_channels':[          # List of the number of channels for each layer of the GAN (starts with input)
                                GAN_CHANNELS*8,
                                GAN_CHANNELS*4,
                                GAN_CHANNELS*2,
                                GAN_CHANNELS,
                                RGB_CHANNELS
                                ],
    'gan_layer_padding':[          # Padding for each layer of the GAN (starts with input)
                                None,
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                (MAIN_PADDING, MAIN_PADDING),
                                ],
    'gan_layer_stride':[          # Stride for each layer of the GAN (starts with input)
                                None,
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                (MAIN_STRIDE, MAIN_STRIDE),
                                ],
    'gan_layer_activation_func':[       # List of the activations functions to be applied to each layer (empty string is no activation function)
                                    '',
                                    'relu',
                                    'relu',
                                    'relu',
                                    'tanh'
                                    ]
    }
