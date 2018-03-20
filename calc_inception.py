from inception_score import *

import os
import os.path
import torch
import skimage.io
import skimage.transform
from sklearn.preprocessing import scale
import numpy as np

import constants

DATA = 'Data/inception_scores'

def load_images():
    filenames = os.listdir(DATA)
    images = []
    for f in filenames:
        image_file = os.path.join(DATA, f)
        curr_image = skimage.io.imread(image_file)
        resized_image = skimage.transform.resize(curr_image,
                            (constants.IMAGE_SIZE, constants.IMAGE_SIZE)).astype('float32')

        img = resized_image.T
        images.append(img)

    images = np.array(images)
    return torch.FloatTensor(images)

imgs = load_images()
print (imgs)

scores = inception_score(imgs, batch_size=5, resize=False, splits=1)
print (scores)
