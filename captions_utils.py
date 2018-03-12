import os
import time
import numpy as np
import constants
import cPickle as pickle
# from numba import jit


# @jit(nopython=True, parallel=True)
def create_caption_dict(data_dir):

	img_dir = os.path.join(data_dir, constants.DIRECTORY_PATH)
	image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
	print (image_files[300:400])
	print (len(image_files))
	image_captions = { img_file : [] for img_file in image_files }

    flowers_caption = os.path.join(data_dir, constants.FLOWERS_CAPTION)
    for i in range(1, constants.FLOWERS_CLASSES + 1):
        class_dir = os.path.join(flowers_caption, 'class_%.5d'%(i))
        for caption_file in os.listdir(class_dir):
            if 'txt' not in caption_file: continue
            with open(join(class_dir, caption_file)) as f:
                captions = f.read().split('\n')
            img_file = caption_file[0:11] + ".jpg"

            if img_file in image_captions:
                image_captions[img_file] += [caption for caption in captions in len(caption) > 0][0:5]

	print (len(image_captions))
    pickle.dump( image_captions, open( os.path.join(data_dir,constants.FLOWERS_CAP_DICT), "wb" ) )
    return image_captions

def load_flowers_capt_dict():
    """Use pickle to load the flowers captions"""
    flowers_capt_dict = pickle.load(open( constants.FLOWERS_CAP_DICT, "rb" ))
    return flowers_capt_dict

def create_coco_capt_dict(data_dir):
    pass

def load_coco_capt_dict():
    coco_capt_dict = pickle.load(open(constants.COCO_CAP_DICT, "rb"))
    return coco_capt_dict

def main():
    data_dir = 'Data'
    create_caption_dict(data_dir)

if __name__ == '__main__':
    main()
