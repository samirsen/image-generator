'''
util.py
'''
import os.path
import h5py
import numpy as np

# Adapted from https://github.com/paarthneekhara/text-to-image
# Takes the directoy and file name of the hdf5 file that contains the word vectors
# Returns a dict from image to caption
def load_text_vec(directory, file_name):
    h = h5py.File(os.path.join(directory, file_name))
    captions = {}
    for item in h.iteritems():
        captions[item[0]] = np.array(item[1])

    return captions
