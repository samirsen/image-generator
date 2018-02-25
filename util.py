'''
util.py
'''
import os.path
import h5py
import numpy as np
import constants
import skimage.io
import skimage.transform

# Adapted from https://github.com/paarthneekhara/text-to-image
# Takes the directoy and file name of the hdf5 file that contains the word vectors
# Returns a dict from image to list of captions
def load_text_vec(directory, file_name):
    h = h5py.File(os.path.join(directory, file_name))
    captions = {}
    for item in h.iteritems():
        captions[item[0]] = np.array(item[1])

    return captions

# Takes in the directory and a list of file names and returns a dict of file name -> images
def load_images(directory, image_file_names):
    image_dict = {}
    for name in image_file_names:
        image_file = os.path.join(directory + name)
        curr_image = skimage.io.imread(image_file)
        # Resize image to correct size as float 32
        resized_image = skimage.transform.resize(curr_image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE)).astype('float32')
        image_dict[name] = resized_image

    return image_dict
