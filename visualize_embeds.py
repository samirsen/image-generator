import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from lstm_model import LSTM
from vocab import get_glove
from data_batcher import *

import constants
import os.path
import h5py

from sklearn.manifold import TSNE
from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt

np.random.seed(42)

COLOR_VEC = ['b', 'g', 'c', 'm', 'y', 'r', 'k']

def load_glove(paths):
    embeddings, word2id, id2word = (torch.load(path) for path in paths)
    return embeddings, word2id, id2word

def lstm_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal(param)

def load_text_vec(directory, file_name):
    """Get N x D numpy matrix of caption embeddings"""

    h = h5py.File(os.path.join(directory, file_name))
    caption_embeddings = []
    for item in h.iteritems():
        name = item[0]
        idx = np.random.randint(len(item[1]))
        caption_embeddings.append(item[1][idx])

    return np.array(caption_embeddings)

def get_embeddings(batch_keys, caption_dict, word2id, lstm):
    captions_batch, masks = get_captions_batch(batch_keys, caption_dict, word2id)
    captions_batch = np.array(captions_batch, dtype=np.int64)



def load_lstm_vecs():
    if os.path.exists('Data/vocab/glove_matrix.torch'):
        paths = ['Data/vocab/glove_matrix.torch', 'Data/vocab/word_to_idx.torch', 'Data/vocab/idx_to_word.torch']
        embeddings, word2id, id2word = load_glove(paths)

    caption_dict = load_flowers_capt_dict(data_dir='Data')

    lstm = LSTM(model_options, embeddings)

    caption_embeddings = []  # 8189 x 300 dimensional
    for i, batch_iter in enumerate(grouper(caption_dict.keys(), constants.BATCH_SIZE)):
        batch_keys = [x for x in batch_iter if x is not None]
        curr_batch_size = len(batch_keys)

        caption_embeds = get_embeddings(batch_keys, caption_dict, word2id, lstm)

        caption_embeddings.append(caption_embeds)

    return np.array(caption_embeddings)



def relabel_embeds(data_dir='Data'):
    labels = []

    caption_dir = os.path.join(data_dir, 'flowers/text_c10')
    classification = 1
    for i in range(1, 103):
        class_dir_name = 'class_%.5d'%(i)
        class_dir = os.path.join(caption_dir, class_dir_name)

        num_images = len(os.listdir(class_dir))
        if num_images % 2 == 0: num_images = num_images / 2
        else: num_images = num_images / 2 + 1

        if i % 17 == 0: classification += 1
        labels.extend([classification] * num_images)

    return np.array(labels)

def get_colors(labels):
    # 17 classes
    colors = []   # should be 8189 dimensional
    for label in labels:
        if label >= len(COLOR_VEC): label = len(COLOR_VEC) - 1
        colors.append(COLOR_VEC[label])

    return colors

print("Visualizing the caption_embeddings ...")
skip_thoughts = load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME)
labels = relabel_embeds()
print skip_thoughts.shape
print labels.shape

# perform t-SNE embedding
vis_data = TSNE(n_components=2).fit_transform(skip_thoughts)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

# get the colors for the different classes
print labels
colors = get_colors(labels)

plt.scatter(vis_x, vis_y, c=colors, alpha=0.5)
# plt.colorbar(ticks=range(102))
# plt.clim(-0.5, 9.5)
plt.show()
