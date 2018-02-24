# image-generator
Generate images from captions using GANs

Karan Singhal, Samir Sen, Trevor Tsue


Go to constants.py and change all of the following to True:
GET_FLOWER_IMAGES 
GET_SKIPTHOUGHT_MODEL
GET_NLTK_PUNKT
GET_PRETRAINED_MODEL

Then run the following to download all of the data:
python download_data.py

Then run the following to generate the skipthought vectors:
python data_loader.py

Then run the main file
python main.py

