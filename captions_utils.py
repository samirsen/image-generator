import os
import time
import numpy as np
import constants
import skimage.io
import skimage.transform
import cPickle as pickle
# from numba import jit
def create_images_dict(filenames):
    img_dict = {}

    flowers_dir = os.path.join('Data', constants.SMALL_DATASET)
    for f in filenames:
        image_path = flowers_dir + f
        curr_image = skimage.io.imread(image_path)
        resized_image = skimage.transform.resize(curr_image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE)).astype('float32')
        img_dict[f] = resized_image

    pickle.dump( (img_dict, open( os.path.join('Data',constants.FLOWERS_IMG_DICT), "wb" ) )
    return img_dict

def load_image_dataset(data_dir='Data'):
	flowers_dir = os.path.join(data_dir,constants.FLOWERS_IMG_DICT)
	flowers_img_dict = pickle.load(open( flowers_dir, "rb" ))
	return flowers_img_dict

# @jit(nopython=True, parallel=True)
def create_caption_dict(data_dir):
	img_dir = os.path.join(data_dir, constants.DIRECTORY_PATH)
	image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
	image_captions = { img_file : [] for img_file in image_files }

	flowers_caption = os.path.join(data_dir, constants.FLOWERS_CAPTION_DIR)
	for i in range(1, constants.FLOWERS_CLASSES + 1):
		class_dir = os.path.join(flowers_caption, 'class_%.5d' % (i))
		for caption_file in os.listdir(class_dir):
			if 'txt' not in caption_file: continue
			with open(os.path.join(class_dir, caption_file)) as f:
				captions = f.read().split('\n')
			img_file = caption_file[0:11] + ".jpg"

			if img_file in image_captions:
				image_captions[img_file] += [caption for caption in captions if len(caption) > 0][:5]

	print (len(image_captions))
	pickle.dump( image_captions, open( os.path.join(data_dir,constants.FLOWERS_CAP_DICT), "wb" ) )
	return image_captions

def load_flowers_capt_dict(data_dir):
    """Use pickle to load the flowers captions"""
    flowers_dir = os.path.join(data_dir,constants.FLOWERS_CAP_DICT)
    flowers_capt_dict = pickle.load(open( flowers_dir, "rb" ))
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
