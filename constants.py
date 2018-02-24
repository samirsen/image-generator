'''
constants.py
Holds all of the constants info of the project
'''

# Three channels for Red, Green, and Blue
RGB_CHANNELS = 3

# One Dimension of an image
IMAGE_SIZE = 1

# The basis for the number of channels in the generator (everything else is a multiple of this)
GAN_CHANNELS = 1

MAIN_MODEL_OPTIONS = {
    'caption_vec_len':1,    # Dimensions for the embedded captions vector
    't_dim':1,              # Dimensions for the text vector inputted into the GAN
    'leak':0.2,             # Leak for Leaky ReLU
    'z_dim':None,
    'batch_size':None,
    'image_size':IMAGE_SIZE,
    'g_channels':GAN_CHANNELS,      # The basis for the number of channels in the generator
    'd1_dim':None,
    'gfc_dim':None,
    'gan_num_layers':4,   # Number of layers in the GAN
    'gan_layer_filter_sizes':[        # List of filter sizes for each layer of the GAN (starts with input)
                                int(IMAGE_SIZE/16),
                                int(IMAGE_SIZE/8),
                                int(IMAGE_SIZE/4),
                                int(IMAGE_SIZE/2),
                                int(IMAGE_SIZE)
                                ],
    'gan_layer_num_channels':[        # List of the number of channels for each layer of the GAN (starts with input)
                                GAN_CHANNELS*8,
                                GAN_CHANNELS*4,
                                GAN_CHANNELS*2,
                                GAN_CHANNELS,
                                RGB_CHANNELS
                                ],
    'gan_layer_activation_func':[     # List of the activations functions to be applied to each layer
                                    '',
                                    'relu',
                                    'relu',
                                    'relu',
                                    'tanh'
                                    ]
    }
