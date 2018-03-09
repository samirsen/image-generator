'''
constants.py
Holds all of the constants info of the project
'''
import torch.nn as nn



'''
Experimental management constants
'''
# Epoch used to declare something as an experiment and output a report
REPORT_EPOCH = 1
EXP_REPORT = "Base model with LR set to 0.0001 and data aug"


'''
data_loader.py constants
'''
# The entire 102 Category Flower Dataset
ENTIRE_DATASET = 'flowers/jpg/'
# A small subset (the first 128 flower pictures) of the entire flower dataset (used for testing code out)
SMALL_DATASET = 'flowers/smalljpg/'

# Choose which directory holds the images you want for the flowers
# TODO: UPDATE THIS ONE FOR WHICH DATA SET YOU WANT
DIRECTORY_PATH = ENTIRE_DATASET

# The name of the output file that the word vectors will be saved as.
# This file name should end with '.hdf5'
VEC_OUTPUT_FILE_NAME = 'flower_tv.hdf5'

'''
Model Constants
main.py, gan_model.py
'''
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
# Learning rate for the Adam Optimizer
LR = 0.0001
# Beta options for the Adam Optimizer
BETAS = (0.5, 0.999)
# Size for each dimension of the image
IMAGE_SIZE = 128


# Options for the main model
MAIN_MODEL_OPTIONS = {
    'caption_vec_len':4800,     # Dimensions for the embedded captions vector
    't_dim':256,                # Dimensions for the text vector input into the GAN
    'z_dim':100,                # Dimensions for the noise vector input into the GAN
    'image_size':IMAGE_SIZE,           # Number of pixels in each dimension of the image
    'num_gf':64,                # Number of generator filters in first layer of generator
    'num_df':64,                # Number of discriminator filters in first layer of discriminator
    'image_channels':3,         # Number of channels for the output of the generator and input of discriminator
    'leak':0.2,                 # Leak for Leaky ReLU
    }
