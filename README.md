# image-generator
Generate images from captions using GANs. Runs a PyTorch implementation.

## Authors
Samir Sen, Trevor Tsue

## Acknowledgements
Adapted parts of code from
https://github.com/paarthneekhara/text-to-image

Borrowed architecture from
https://github.com/reedscot/icml2016


## Getting started
Run the following to download all of the data:
```
python download_data.py
```

Then run the following to generate the skipthought vectors:
```
python data_loader.py
```


## Training the Data
Run the main file to start training the GAN
```
python main.py
```
