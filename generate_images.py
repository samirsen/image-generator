import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import constants


def generate(sentence, model_weights_path, save_path, use_gpu=False, num_images=10):
	if constants.USE_MODEL == 'began':
	    generator = BeganGenerator(model_options)
	    discriminator = BeganDiscriminator(model_options)
	elif constants.USE_MODEL == 'wgan':
	    generator = WGanGenerator(model_options)
	    discriminator = WGanDiscriminator(model_options)
	else:
	    generator = Generator(model_options)
	    discriminator = Discriminator(model_options)

	generator.eval()
	discriminator.eval()

	

	if torch.cuda.is_available() and use_gpu:
	    print("CUDA is available")
	    generator = generator.cuda()
	    discriminator = discriminator.cuda()
	    print("Moved models to GPU")



