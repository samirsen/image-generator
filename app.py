import flask
from flask import Flask, jsonify, request, render_template
import sys
import skipthoughts 
from demo_model import CondBeganGenerator, CondBeganDiscriminator
import torch
import scipy.misc
import constants
from torch.autograd import Variable
import numpy as np
import nltk
nltk.download('punkt')

app = Flask(__name__)

WEIGHTS_EPOCH = 1140 
BATCH_SIZE = 32

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	queries = request.get_json(silent=True, force=True)['input']
	# query = "This is a red flower with yellow stamen."
	encoded = Variable(torch.Tensor(skipthoughts.encode(model, queries)))
	if torch.cuda.is_available():
		encoded = encoded.cuda()
	image_paths = []

	for batch_i in range(BATCH_SIZE):
		noise_vec = Variable(torch.randn(len(queries), 100, 1, 1))
		if torch.cuda.is_available():
			noise_vec = noise_vec.cuda()

		gen_images = generator.forward(encoded, noise_vec)
		gen_images = gen_images.cpu()

		for i, img in enumerate(gen_images):
			curr = img.data.numpy()
			curr = np.swapaxes(curr, 0, 1)
			curr = np.swapaxes(curr, 1, 2)
			path = 'Data/samples/' + str(batch_i) + '_' + str(i) + '.png'
			scipy.misc.imsave(path, curr)
			image_paths.append(path)
	

	return jsonify({'images': image_paths})



if __name__ == '__main__':
	

	model_options = {
    'verbose':True,   # Prints out info about the model
    'caption_vec_len':4800,         # Dimensions for the embedded captions vector
    't_dim':256,                    # Dimensions for the text vector input into the GAN
    'z_dim':100,                    # Dimensions for the noise vector input into the GAN
    'image_size':128,           # Number of pixels in each dimension of the image
    'num_gf':64,                    # Number of generator filters in first layer of generator
    'num_df':64,                    # Number of discriminator filters in first layer of discriminator
    'image_channels':3,             # Number of channels for the output of the generator and input of discriminator
    'leak':0.2,                     # Leak for Leaky ReLU
    'label_smooth':0.1,             # One-sided label smoothing for the real labels
                                    # e.g. with label_smooth of 0.1, instead of real label = 1, we have real_label = 1 - 0.1
                                    # https://arxiv.org/pdf/1606.03498.pdf
    # CLS (Conditional Loss Sensitivity) Options
    'use_cls':True,
    # WGAN Options
    'wgan_d_iter':5,                # Number of times to train D before training G
    # BEGAN OPTIONS
    'began_gamma':0.5,              # Gamma value for BEGAN model (balance between D and G)
    'began_lambda_k':0.001,         # Learning rate for k of BEGAN model
    'began_hidden_size':64,         # Hidden size for embedder of BEGAN model
    }
	generator = CondBeganGenerator(model_options)
	discriminator = CondBeganDiscriminator(model_options)

	print("Created generator and discriminator")

	if torch.cuda.is_available():
	    print("CUDA is available")
	    generator = generator.cuda()
	    discriminator = discriminator.cuda()
	    print("Moved models to GPU")

	if torch.cuda.is_available():
		checkpoint = torch.load('Data/Models/began-final/weights/epoch' + str(WEIGHTS_EPOCH))
	else:
		checkpoint = torch.load('Data/Models/began-final/weights/epoch' + str(WEIGHTS_EPOCH), map_location=lambda storage, loc: storage)
	generator.load_state_dict(checkpoint['g_dict'])
	discriminator.load_state_dict(checkpoint['d_dict'])
	discriminator.began_k = checkpoint['began_k']


	generator.train()
	discriminator.train()

	model = skipthoughts.load_model()


	app.run(host='0.0.0.0', port=8000, debug=True)