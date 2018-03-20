import flask
from flask import Flask, jsonify, request, render_template
import sys
sys.path.append("..")
import skipthoughts 
from demo_model import Generator, Discriminator
import torch
import scipy.misc
import constants
from torch.autograd import Variable
import numpy as np
import nltk
nltk.download('punkt')

app = Flask(__name__)

WEIGHTS_EPOCH = 520 
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
	noise_vec = Variable(torch.randn(len(queries), 100, 1, 1))
	for batch_i in range(BATCH_SIZE):
		
		if torch.cuda.is_available():
			noise_vec = noise_vec.cuda()

		gen_images = generator.forward(encoded, noise_vec)
		gen_images = gen_images.cpu()

		for i, img in enumerate(gen_images):
			curr = gen_images[0].data.numpy()
			curr = np.swapaxes(curr, 0, 1)
			curr = np.swapaxes(curr, 1, 2)
			path = 'Data/samples/' + str(batch_i) + '_' + str(i) + '.png'
			scipy.misc.imsave(path, curr)
			image_paths.append(path)
	

	return jsonify({'images': image_paths})



if __name__ == '__main__':
	

	# Options for the main model
	model_options = {
	    'caption_vec_len':4800,     # Dimensions for the embedded captions vector
	    't_dim':256,                # Dimensions for the text vector input into the GAN
	    'z_dim':100,                # Dimensions for the noise vector input into the GAN
	    'image_size':128,           # Number of pixels in each dimension of the image
	    'num_gf':64,                # Number of generator filters in first layer of generator
	    'num_df':64,                # Number of discriminator filters in first layer of discriminator
	    'image_channels':3,         # Number of channels for the output of the generator and input of discriminator
	    'leak':0.2,                 # Leak for Leaky ReLU
	    }
	generator = Generator(model_options)
	discriminator = Discriminator(model_options)

	print("Created generator and discriminator")

	if torch.cuda.is_available():
	    print("CUDA is available")
	    generator = generator.cuda()
	    discriminator = discriminator.cuda()
	    print("Moved models to GPU")



	if torch.cuda.is_available():
		gen_state = torch.load('Data/Models/baseline/weights/g_epoch' + str(WEIGHTS_EPOCH))
		dis_state = torch.load('Data/Models/baseline/weights/d_epoch' + str(WEIGHTS_EPOCH))
	else:
		gen_state = torch.load('Data/Models/baseline/weights/g_epoch' + str(WEIGHTS_EPOCH), map_location=lambda storage, loc: storage)
		dis_state = torch.load('Data/Models/baseline/weights/d_epoch' + str(WEIGHTS_EPOCH), map_location=lambda storage, loc: storage)

	generator.load_state_dict(gen_state)
	discriminator.load_state_dict(dis_state)



	generator.train()
	discriminator.train()

	model = skipthoughts.load_model()


	app.run(host='0.0.0.0', port=8000, debug=True)