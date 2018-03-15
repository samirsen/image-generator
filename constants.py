'''
constants.py
Holds all of the constants info of the project
'''



'''
Experimental management constants
'''
# Epoch used to declare something as an experiment and output a report
REPORT_EPOCH = 1
EXP_REPORT = "Base model with LR set to 0.0001 and data aug and requires_grad=False for inputs"


'''
data_loader.py constants
'''
# The entire 102 Category Flower Dataset
ENTIRE_DATASET = 'flowers/jpg/'
# A small subset (the first 128 flower pictures) of the entire flower dataset (used for testing code out)
SMALL_DATASET = 'flowers/smalljpg/'
# Directory the Flowers captions are located
FLOWERS_CAPTION_DIR = 'flowers/text_c10'
# Flowers dictionary save file
FLOWERS_CAP_DICT = "flowers_captions.p"
# Number of flower classes in the dataset
FLOWERS_CLASSES = 102

# Choose which directory holds the images you want for the flowers
# NOTE: UPDATE THIS ONE FOR WHICH DATA SET YOU WANT
DIRECTORY_PATH = ENTIRE_DATASET

# The name of the output file that the word vectors will be saved as.
# This file name should end with '.hdf5'
VEC_OUTPUT_FILE_NAME = 'flower_tv.hdf5'

'''
Model Constants
main.py
'''
##### MODEL OPTIONS ####
# Generate another image for the training of the G (don't use the one from D)
# Most models do regenerate image (however, we did not for our baseline)
REGEN_IMAGE = True
# Conditional Loss Sensitivity (CLS)
# Add the option of penalizing GAN for matching image with wrong caption
USE_CLS = False

# The different models to use
# 'dcgan', 'wgan', 'began'
USE_MODEL = 'began'

##### END MODEL OPTIONS #####



FLOWERS_DICTS_PATH = 'Data/flowers_dicts.torch'
# SAVE PATH FOR MODLE OPTIONS
SAVE_PATH = 'Data/outputs/'
# If true, prints status of creating model
PRINT_MODEL_STATUS = True


# TRAINING OPTIONS
# Number of epochs to run the training
NUM_EPOCHS = 1000
# The batch size for training
BATCH_SIZE = 128
# How often to save losses
LOSS_SAVE_IDX = 1


# Optimizer options
# True if optimizer will be stochastic gradient descent
# False if optimizer will be adam
D_OPTIMIZER_SGD = False
# Learning rate for the Optimizer
LR = 0.0001
# Beta options for the Adam Optimizer
BETAS = (0.5, 0.999)
# The learning decays after this many iterations
LR_DECAY_EVERY = 3000


# Size for each dimension of the image
IMAGE_SIZE = 128
# Size of GloVe Embeddings
EMBED_DIM = 300
# Size of hidden dim for LSTM
HIDDEN_DIM = EMBED_DIM
# Average or sum of glove vectors
REDUCE_TYPE = 'mean'
# Number of embeddings in Glove
VOCAB_SIZE = 400000



# Options for the main model
MAIN_MODEL_OPTIONS = {
    'verbose':PRINT_MODEL_STATUS,   # Prints out info about the model
    'caption_vec_len':4800,         # Dimensions for the embedded captions vector
    't_dim':256,                    # Dimensions for the text vector input into the GAN
    'z_dim':100,                    # Dimensions for the noise vector input into the GAN
    'image_size':IMAGE_SIZE,           # Number of pixels in each dimension of the image
    'num_gf':64,                    # Number of generator filters in first layer of generator
    'num_df':64,                    # Number of discriminator filters in first layer of discriminator
    'image_channels':3,             # Number of channels for the output of the generator and input of discriminator
    'leak':0.2,                     # Leak for Leaky ReLU
    'label_smooth':0.1,             # One-sided label smoothing for the real labels
                                    # e.g. with label_smooth of 0.1, instead of real label = 1, we have real_label = 1 - 0.1
                                    # https://arxiv.org/pdf/1606.03498.pdf
    # CLS (Conditional Loss Sensitivity) Options
    'use_cls':USE_CLS,
    # WGAN Options
    'wgan_d_iter':5,                # Number of times to train D before training G
    # BEGAN OPTIONS
    'began_gamma':0.5,              # Gamma value for BEGAN model (balance between D and G)
    'began_lambda_k':0.001,         # Learning rate for k of BEGAN model
    'began_hidden_size':64,         # Hidden size for embedder of BEGAN model
    }
