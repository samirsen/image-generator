'''
util.py
'''
import os.path
import h5py
import numpy as np
import constants
import skimage.io
import skimage.transform
from scipy.io import loadmat
import glob
import os
import cPickle as pickle
import torch


# Makes the directories of they don't already exist
def make_directories():
    output_path = constants.SAVE_PATH
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Made output directory")
    else:
        print("WARNING: starting training with an existing outputs directory")
    if not os.path.exists(output_path + 'weights/'):
        os.makedirs(output_path + 'weights/')
        print("Made weights directory")
    if not os.path.exists(output_path + 'images/'):
        os.makedirs(output_path + 'images/')
        print("Made images directory")


# Loads a map from image file names to 'test', 'train', or 'val'
# Used in other functions to split data
def load_dataset_map():
    ids = loadmat('data_constants/setid.mat')
    # Flip train and test examples since otherwise there would be 6000 test
    train_ids = ids['tstid'][0] - 1
    test_ids = ids['trnid'][0] - 1
    val_ids = ids['valid'][0] - 1

    print(len(train_ids), len(val_ids), len(test_ids), "Train, val, test examples, respectively")
    filenames = [name for name in os.listdir('Data/' + constants.ENTIRE_DATASET) if name.endswith('.jpg')]
    image_paths = sorted(filenames)
    dataset_map = {}
    for i, name in enumerate(image_paths):
        if i in train_ids:
            dataset_map[name] = 'train'
        elif i in test_ids:
            dataset_map[name] ='test'
        elif i in val_ids:
            dataset_map[name] ='val'
        else:
            print("Invalid ID!")
    return dataset_map


def load_flowers_capt_dict():
    """Use pickle to load the flowers captions"""
    flowers_capt_dict = pickle.load(open( constants.FLOWERS_CAP_DICT, "rb" ))
    return flowers_capt_dict

def load_coco_capt_dict():
    """Use pickle to load the MSCOCO captions"""
    coco_capt_dict = pickle.load(open(constants.COCO_CAP_DICT, "rb"))
    return coco_capt_dict


# Adapted from https://github.com/paarthneekhara/text-to-image
# Takes the directoy and file name of the hdf5 file that contains the word vectors
# Returns a dict from image to list of captions
def load_text_vec(directory, file_name, dataset_map):
    h = h5py.File(os.path.join(directory, file_name))
    train_captions, val_captions, test_captions = {}, {}, {}
    for item in h.iteritems():
        name = item[0]
        if dataset_map[name] == 'train':
            train_captions[name] = np.array(item[1])
        elif dataset_map[name] =='val':
            val_captions[name] = np.array(item[1])
        elif dataset_map[name] =='test':
            test_captions[name] = np.array(item[1])
        else:
            print("Invalid name")

    return train_captions, val_captions, test_captions

# Gets images for the main function
def get_images(directory, file_name, save_path):
    if os.path.exists(save_path):
        image_dicts = torch.load(save_path)
        train_image_dict, val_image_dict, test_image_dict = image_dicts
        print("Loaded images")
    else:
        print("Loading images and separating into train/val/test sets")
        path = os.path.join(directory, file_name)
        filenames = train_captions.keys() + val_captions.keys() + test_captions.keys()
        train_image_dict, val_image_dict, test_image_dict = util.load_images(path, filenames, dataset_map)
        image_dicts = [train_image_dict, val_image_dict, test_image_dict]
        torch.save(image_dicts, save_path)

    return train_image_dict, val_image_dict, test_image_dict


# Takes in the directory and a list of file names and returns a dict of file name -> images
def load_images(directory, filenames, dataset_map):
    train_image_dict, val_image_dict, test_image_dict = {}, {}, {}
    for name in filenames:
        image_file = os.path.join(directory + name)
        curr_image = skimage.io.imread(image_file)
        # Resize image to correct size as float 32
        resized_image = skimage.transform.resize(curr_image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE)).astype('float32')

        if dataset_map[name] =='train':
            train_image_dict[name] = resized_image
        elif dataset_map[name] =='val':
            val_image_dict[name] = resized_image
        elif dataset_map[name] =='test':
            test_image_dict[name] = resized_image
        else:
            print("Invalid name")


    return train_image_dict, val_image_dict, test_image_dict


# custom weights initialization called on netG and netD
# from https://github.com/pytorch/examples/blob/master/dcgan/main.py
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def preprocess2(batch_input):
    """Inputs for self.embeddings in TextModel(). Batch_input must be numpy padded"""
    batch_size, sent_len = batch_input.shape
    offsets = [sent_len * i for i in range(batch_size)]
    return batch_input.flatten(), offsets


def preprocess(batch_input):
    """If batch_input isn't numpy"""
    flatten, offsets = [], []
    index = 0
    for ex in batch_input:
        offsets.append(index)
        flatten.extend(ex)
        index += len(ex)

    return flatten, offsets

# https://github.com/sunshineatnoon/Paper-Implementations/blob/master/BEGAN/began.py
def adjust_learning_rate(optimizer, niter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = constants.LR * (0.95 ** (niter // constants.LR_DECAY_EVERY))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
