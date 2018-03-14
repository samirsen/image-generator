'''
model.py
Based on the models from
	https://github.com/pytorch/examples/blob/master/dcgan/main.py
	https://github.com/paarthneekhara/text-to-image
	https://github.com/aelnouby/Text-to-Image-Synthesis/tree/master/models
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import constants
import collections
import functools
from util import *

'''
	OPTIONS
	verbose : Prints out info about the model
	caption_vector_length : Caption Vector Length 2400
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	num_gf : Number of generator filters in first layer of generator
	num_df : Number of discriminator filters in first layer of discriminator
	image_channels: Number of channels for the output of the generator and input of discriminator
					Usually, 3 channels because of RGB.
	leak : Leak for Leaky ReLU
	label_smooth : One-sided label smoothing for the real labels
    began_gamma : Gamma value for BEGAN model (balance between D and G)
    began_lambda_k : Learning rate for k of BEGAN model
'''



'''
DCGAN Model
'''
class Generator(nn.Module):
	def __init__(self, options):
		super(Generator, self).__init__()

		self.options = options
		# Dimensions of the latent vector (concatenate processed embedding vector and noise vector)
		self.options['concat_dim'] = self.options['t_dim'] + self.options['z_dim']

		if self.options['verbose']: print('\nCreating Generator...')

		# Projector processes the word embedding before we concatenate embedding with noise
		self.g_projector = nn.Sequential(
			nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if self.options['verbose']: print('Generator Projector Created')

		# Generator inputs concated word embedding and noise vector (latent vector) and outputs image
		self.generator = nn.Sequential(
			# Input Dim: batch_size x (concat_dim) x 1 x 1
			nn.ConvTranspose2d(self.options['concat_dim'], self.options['num_gf'] * 16, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 16),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 16) x 4 x 4
			nn.ConvTranspose2d(self.options['num_gf'] * 16, self.options['num_gf'] * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 8),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 8) x 8 x 8
			nn.ConvTranspose2d(self.options['num_gf'] * 8, self.options['num_gf'] * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 4),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 4) x 16 x 16
			nn.ConvTranspose2d(self.options['num_gf'] * 4, self.options['num_gf'] * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 2),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 2) x 32 x 32
			nn.ConvTranspose2d(self.options['num_gf'] * 2, self.options['num_gf'], 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf) x 64 x 64
			nn.ConvTranspose2d(self.options['num_gf'], self.options['image_channels'], 4, 2, 1, bias=False),
			nn.Tanh()
			# Dim: batch_size x (num_channels) x 128 x 128
		)

		if self.options['verbose']: print('Generator Created\n')


	# Generator Forward Propagation
	def forward(self, text_embed, noise):
		projected_embed = self.g_projector(text_embed)
		# Add dimension 2 and 3 to make projected embed into 4 dimension
		# batch_size x num_channels x height (1) x width (1)
		projected_embed = projected_embed.unsqueeze(2).unsqueeze(3)
		latent_vec = torch.cat([projected_embed, noise], 1)
		output = self.generator(latent_vec)

		return output


	# Generator Loss
	# L_G = log(y_f)
	def loss(self, fake_img_passed):
		g_loss = f.binary_cross_entropy(fake_img_passed, torch.ones_like(fake_img_passed))

		return g_loss

	# Calculates the grad of g
	def calc_grad_g(self, new_fake_img_passed):
		g_loss = self.loss(new_fake_img_passed)

		return g_loss

class Discriminator(nn.Module):
	def __init__(self, options):
		super(Discriminator, self).__init__()

		self.options = options

		if self.options['verbose']: print('Creating Discriminator...')

		# Discriminator layers for the input of the image
		self.discriminator_input = nn.Sequential(
			# Input Dim: batch_size x (num_channels) x 128 x 128
			nn.Conv2d(self.options['image_channels'], self.options['num_df'], 4, 2, 1, bias=False),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df) x 64 x 64
			nn.Conv2d(self.options['num_df'], self.options['num_df'] * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 2),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 2) x 32 x 32
			nn.Conv2d(self.options['num_df'] * 2, self.options['num_df'] * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 4),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 4) x 16 x 16
			nn.Conv2d(self.options['num_df'] * 4, self.options['num_df'] * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 8),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 8) x 8 x 8
			nn.Conv2d(self.options['num_df'] * 8, self.options['num_df'] * 16, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 16),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 16) x 4  x 4
		)

		if self.options['verbose']: print('Discriminator Input Created')

		# Discriminator layers for the projection of the text embedding
		self.d_projector = nn.Sequential(
		    nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if self.options['verbose']: print('Discriminator Projector Created')

		# Discriminator layers for the concatenation of the text embedding and image
		# Vanilla GAN uses sigmoid output
		self.discriminator_output = nn.Sequential(
			# Dim: batch_size x (num_df * 16 + t_dim) x 4 x 4
			nn.Conv2d(self.options['num_df'] * 16 + self.options['t_dim'], 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			# Dim: batch_size x 1 x 1 x 1
		)

		if self.options['verbose']: print('Discriminator Output Created')
		if self.options['verbose']: print('Discriminator Created\n')


	# Discriminator Forward Propagation
	def forward(self, images, text_embed):
		images_intermediate = self.discriminator_input(images)
		projected_embed = self.d_projector(text_embed)
		# Repeat the projected dimensions and change the permutations
		# Dim: batch_size x 256 -> batch_size x 256 x 4 x 4
		replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
		latent_vec = torch.cat([images_intermediate, replicated_embed], 1)
		output = self.discriminator_output(latent_vec)
		# Squeeze dims: batch_size x 1 x 1 x 1 -> batch_size
		output = output.view(-1, 1).squeeze(1)

		return output


	# Overall loss function for discriminator
	# L_D = log(y_r) + log(1 - y_f)
	# Loss of Vanilla GAN with CLS
	# log(1 - y_w) is the caption loss sensitivity CLS (makes sure that captions match the image)
	# L_D = log(y_r) + log(1 - y_w) + log(1 - y_f)
	def loss(self, real_img_passed, fake_img_passed, wrong_img_passed=None):
		# Add one-sided label smoothing to the real images of the discriminator
		d_real_loss = f.binary_cross_entropy(real_img_passed, torch.ones_like(real_img_passed) - self.options['label_smooth'])
		d_fake_loss = f.binary_cross_entropy(fake_img_passed, torch.zeros_like(fake_img_passed))

		d_loss = d_real_loss + d_fake_loss

		# option to use conditional loss sensitivity
		if self.options['use_cls']:
			d_wrong_loss = f.binary_cross_entropy(wrong_img_passed, torch.zeros_like(wrong_img_passed))
			d_loss += d_wrong_loss


		return d_loss

	# Calculates the gradients for the D and returns the D loss
	def calc_grad_d(self, real_img_passed, fake_img_passed, wrong_img_passed=None):
	    if self.options['use_cls']:
	        d_loss = self.loss(real_img_passed, fake_img_passed, wrong_img_passed)
	    else:
	        d_loss = self.loss(real_img_passed, fake_img_passed)

	    d_loss.backward()

	    return d_loss


'''
WGAN Model
Based on paper https://arxiv.org/pdf/1701.07875.pdf
https://github.com/martinarjovsky/WassersteinGAN
'''
class WGanGenerator(nn.Module):
	def __init__(self, options):
		super(WGanGenerator, self).__init__()

		self.options = options
		# Dimensions of the latent vector (concatenate processed embedding vector and noise vector)
		self.options['concat_dim'] = self.options['t_dim'] + self.options['z_dim']

		# Grad factor alters whether we step in positive direction (grad_factor = 1) or negative (neg_grad_factor = -1)
		self.grad_factor = Variable(torch.Tensor([1]))
		self.neg_grad_factor = Variable(torch.Tensor([-1]))
		if torch.cuda.is_available():
			self.grad_factor = self.grad_factor.cuda()
			self.neg_grad_factor = self.neg_grad_factor.cuda()

		if self.options['verbose']: print('\nCreating WGAN Generator...')

		# Projector processes the word embedding before we concatenate embedding with noise
		self.g_projector = nn.Sequential(
			nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if self.options['verbose']: print('WGAN Generator Projector Created')

		# Generator inputs concated word embedding and noise vector (latent vector) and outputs image
		self.generator = nn.Sequential(
			# Input Dim: batch_size x (concat_dim) x 1 x 1
			nn.ConvTranspose2d(self.options['concat_dim'], self.options['num_gf'] * 16, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 16),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 16) x 4 x 4
			nn.ConvTranspose2d(self.options['num_gf'] * 16, self.options['num_gf'] * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 8),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 8) x 8 x 8
			nn.ConvTranspose2d(self.options['num_gf'] * 8, self.options['num_gf'] * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 4),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 4) x 16 x 16
			nn.ConvTranspose2d(self.options['num_gf'] * 4, self.options['num_gf'] * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 2),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf * 2) x 32 x 32
			nn.ConvTranspose2d(self.options['num_gf'] * 2, self.options['num_gf'], 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_gf) x 64 x 64
			nn.ConvTranspose2d(self.options['num_gf'], self.options['image_channels'], 4, 2, 1, bias=False),
			nn.Tanh()
			# Dim: batch_size x (num_channels) x 128 x 128
		)

		if self.options['verbose']: print('WGAN Generator Created\n')

	# WGAN Generator Forward Propagation
	def forward(self, text_embed, noise):
		projected_embed = self.g_projector(text_embed)
		# Add dimension 2 and 3 to make projected embed into 4 dimension
		# batch_size x num_channels x height (1) x width (1)
		projected_embed = projected_embed.unsqueeze(2).unsqueeze(3)
		latent_vec = torch.cat([projected_embed, noise], 1)
		output = self.generator(latent_vec)

		return output


	# WGAN Generator Loss
	# L_G = L(y_f)
	def loss(self, fake_img_passed):
		g_loss = fake_img_passed.mean()

		return g_loss

	# Calculates the grad of g
	def calc_grad_g(self, new_fake_img_passed):
		g_loss = self.loss(new_fake_img_passed)
		g_loss.backward(self.neg_grad_factor)

		return g_loss


class WGanDiscriminator(nn.Module):
	def __init__(self, options):
		super(WGanDiscriminator, self).__init__()

		self.options = options
	    # Grad factor alters whether we step in positive direction (grad_factor = 1) or negative (neg_grad_factor = -1)
		self.grad_factor = Variable(torch.Tensor([1]))
		self.neg_grad_factor = Variable(torch.Tensor([-1]))
		if torch.cuda.is_available():
			self.grad_factor = self.grad_factor.cuda()
			self.neg_grad_factor = self.neg_grad_factor.cuda()

		if self.options['verbose']: print('Creating WGAN Discriminator...')

		# Discriminator layers for the input of the image
		self.discriminator_input = nn.Sequential(
			# Input Dim: batch_size x (num_channels) x 128 x 128
			nn.Conv2d(self.options['image_channels'], self.options['num_df'], 4, 2, 1, bias=False),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df) x 64 x 64
			nn.Conv2d(self.options['num_df'], self.options['num_df'] * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 2),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 2) x 32 x 32
			nn.Conv2d(self.options['num_df'] * 2, self.options['num_df'] * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 4),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 4) x 16 x 16
			nn.Conv2d(self.options['num_df'] * 4, self.options['num_df'] * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 8),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 8) x 8 x 8
			nn.Conv2d(self.options['num_df'] * 8, self.options['num_df'] * 16, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_df'] * 16),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True),
			# Dim: batch_size x (num_df * 16) x 4  x 4
		)

		if self.options['verbose']: print('WGAN Discriminator Input Created')

		# Discriminator layers for the projection of the text embedding
		self.d_projector = nn.Sequential(
		    nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if self.options['verbose']: print('WGAN Discriminator Projector Created')

		# Discriminator layers for the concatenation of the text embedding and image
		# Vanilla GAN uses sigmoid output
		# WGAN does not use sigmoid output
		self.discriminator_output = nn.Sequential(
			# Dim: batch_size x (num_df * 16 + t_dim) x 4 x 4
			nn.Conv2d(self.options['num_df'] * 16 + self.options['t_dim'], 1, 4, 1, 0, bias=False),
			# Dim: batch_size x 1 x 1 x 1
		)

		if self.options['verbose']: print('WGAN Discriminator Output Created')
		if self.options['verbose']: print('WGAN Discriminator Created\n')


	# WGAN Discriminator Forward Propagation
	def forward(self, images, text_embed):
		images_intermediate = self.discriminator_input(images)
		projected_embed = self.d_projector(text_embed)
		# Repeat the projected dimensions and change the permutations
		# Dim: batch_size x 256 -> batch_size x 256 x 4 x 4
		replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
		latent_vec = torch.cat([images_intermediate, replicated_embed], 1)
		output = self.discriminator_output(latent_vec)
		# Squeeze dims: batch_size x 1 x 1 x 1 -> batch_size
		output = output.view(-1, 1).squeeze(1)

		return output


	# Loss of WGAN Discriminator
	# L_D = L(y_r) - L(y_f)
	# Loss of WGAN with CLS (caption loss sensitivity - makes sure captions match the image)
	# L_D = L(y_r) - L(y_w) - L(y_f)
	def loss(self, real_img_passed, fake_img_passed, wrong_img_passed=None):
		d_real_loss = real_img_passed.mean()
		d_fake_loss = fake_img_passed.mean()

		d_loss = d_real_loss - d_fake_loss

		# option to use conditional loss sensitivity
		if self.options['use_cls']:
			d_wrong_loss = wrong_img_passed.mean()
			d_loss -= d_wrong_loss
			return d_loss, d_real_loss, d_fake_loss, d_wrong_loss

		return d_loss, d_real_loss, d_fake_loss


	# Calculate the gradient for the D and returns D loss
	def calc_grad_d(self, real_img_passed, fake_img_passed, wrong_img_passed=None):
		if self.options['use_cls']:
			d_loss, d_real_loss, d_fake_loss, d_wrong_loss = self.loss(real_img_passed, fake_img_passed, wrong_img_passed)
			d_wrong_loss.backward(self.grad_factor)
		else:
			d_loss, d_real_loss, d_fake_loss = self.loss(real_img_passed, fake_img_passed)
		d_real_loss.backward(self.neg_grad_factor)
		d_fake_loss.backward(self.grad_factor)

		return d_loss


'''
BEGAN Model
Based on paper
https://arxiv.org/pdf/1703.10717.pdf
https://github.com/sunshineatnoon/Paper-Implementations/blob/master/BEGAN/models.py
https://github.com/carpedm20/BEGAN-pytorch
'''

def conv_block(input_dim, output_dim):
	return nn.Sequential(
		nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
		nn.ELU(inplace=True),
		nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
		nn.ELU(inplace=True),
		nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
		nn.AvgPool2d(kernel_size=2, stride=2)
	)

# Convolution and upsample doubles size of image (instead of convtranspose)
def upsample_conv_block(input_dim, output_dim, scale):
	return nn.Sequential(
		nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
		nn.ELU(inplace=True),
		nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
		nn.ELU(inplace=True),
		nn.Upsample(scale_factor=scale)
	)

# Unlike the other generator (which uses convtranpse), this generator uses conv and upsampling blocks
class BeganGenerator(nn.Module):
	def __init__(self, options):
		super(BeganGenerator, self).__init__()

		self.options = options
		# Dimensions of the latent vector (concatenate processed embedding vector and noise vector)
		self.options['concat_dim'] = self.options['t_dim'] + self.options['z_dim']

		if self.options['verbose']: print('\nCreating BEGAN Generator...')

		# Projector processes the word embedding before we concatenate embedding with noise
		self.g_projector = nn.Sequential(
			nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if self.options['verbose']: print('BEGAN Generator Projector Created')

		# Generator inputs concated word embedding and noise vector (latent vector) and outputs image
		self.generator = nn.Sequential(
			# Input Dim: batch_size x (concat_dim) x 1 x 1
			upsample_conv_block(self.options['concat_dim'], self.options['num_gf'] * 16, 4),
			# Dim: batch_size x (num_gf * 16) x 4 x 4
			upsample_conv_block(self.options['num_gf'] * 16, self.options['num_gf'] * 8, 2),
			# Dim: batch_size x (num_gf * 8) x 8 x 8
			upsample_conv_block(self.options['num_gf'] * 8, self.options['num_gf'] * 4, 2),
			# Dim: batch_size x (num_gf * 4) x 16 x 16
			upsample_conv_block(self.options['num_gf'] * 4, self.options['num_gf'] * 2, 2),
			# Dim: batch_size x (num_gf * 2) x 32 x 32
			upsample_conv_block(self.options['num_gf'] * 2, self.options['num_gf'], 2),
			# Dim: batch_size x (num_gf) x 64 x 64
			upsample_conv_block(self.options['num_gf'], self.options['num_gf'], 2),
			# Dim: batch_size x (num_gf) x 128 x 128
			nn.Conv2d(self.options['num_gf'], self.options['num_gf'], kernel_size=3, stride=1, padding=1),
			nn.ELU(inplace=True),
			# Dim: batch_size x (num_gf) x 128 x 128
			nn.Conv2d(self.options['num_gf'], self.options['num_gf'], kernel_size=3, stride=1, padding=1),
			nn.ELU(inplace=True),
			# Dim: batch_size x (num_gf) x 128 x 128
			nn.Conv2d(self.options['num_gf'], self.options['image_channels'], kernel_size=3, stride=1, padding=1),
			# Dim: batch_size x (num_image_channels) x 128 x 128
			nn.Tanh()
		)

		if self.options['verbose']: print('BEGAN Generator Created\n')


	# Generator Forward Propagation
	def forward(self, text_embed, noise):
		projected_embed = self.g_projector(text_embed)
		# Add dimension 2 and 3 to make projected embed into 4 dimension
		# batch_size x num_channels x height (1) x width (1)
		projected_embed = projected_embed.unsqueeze(2).unsqueeze(3)
		latent_vec = torch.cat([projected_embed, noise], 1)
		output = self.generator(latent_vec)

		return output

	# Generator Loss
	# L_G = L(y_f)
	def loss(self, fake_img, fake_img_recons):
		g_loss = torch.mean(torch.abs(fake_img_recons - fake_img))

		return g_loss

	# Calculates the grad of g
	def calc_grad_g(self, fake_img, fake_img_recons):
		g_loss = self.loss(fake_img, fake_img_recons)

		return g_loss


class BeganDiscriminator(nn.Module):
	def __init__(self, options):
		super(BeganDiscriminator, self).__init__()

		self.options = options
		# Initialize began k value to 0
		self.began_k = 0

		if self.options['verbose']: print('Creating BEGAN Discriminator...')

		# Discriminator layers for the input of the image
		self.discriminator_input = nn.Sequential(
			# Input Dim: batch_size x (num_channels) x 128 x 128
			nn.Conv2d(self.options['image_channels'], self.options['num_df'], 3, 1, 1),
			nn.ELU(inplace=True),
			conv_block(self.options['num_df'], self.options['num_df']),
			# Dim: batch_size x (num_df) x 64 x 64
			conv_block(self.options['num_df'], self.options['num_df'] * 2),
			# Dim: batch_size x (num_df * 2) x 32 x 32
			conv_block(self.options['num_df'] * 2, self.options['num_df'] * 4),
			# Dim: batch_size x (num_df * 4) x 16 x 16
			conv_block(self.options['num_df'] * 4, self.options['num_df'] * 8),
			# Dim: batch_size x (num_df * 8) x 8 x 8
			conv_block(self.options['num_df'] * 8, self.options['num_df'] * 16),
			# Dim: batch_size x (num_df * 16) x 4 x 4
		)

		if self.options['verbose']: print('BEGAN Discriminator Input Created')

		# Discriminator layers for the projection of the text embedding
		self.d_projector = nn.Sequential(
		    nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if self.options['verbose']: print('BEGAN Discriminator Projector Created')

		# Discriminator upsample layers for the concatenation of the text embedding and image to output an image
		# Reconstructs the image
		self.discriminator_output = nn.Sequential(
			# Input Dim: batch_size x (num_df * 16 + t_dim) x 4 x 4
			upsample_conv_block(self.options['num_gf'] * 16 + self.options['t_dim'], self.options['num_gf'] * 16, 2),
			# Dim: batch_size x (num_gf * 16) x 8 x 8
			upsample_conv_block(self.options['num_gf'] * 16, self.options['num_gf'] * 8, 2),
			# Dim: batch_size x (num_gf * 8) x 16 x 16
			upsample_conv_block(self.options['num_gf'] * 8, self.options['num_gf'] * 4, 2),
			# Dim: batch_size x (num_gf * 4) x 32 x 32
			upsample_conv_block(self.options['num_gf'] * 4, self.options['num_gf'] * 2, 2),
			# Dim: batch_size x (num_gf * 2) x 64 x 64
			upsample_conv_block(self.options['num_gf'] * 2, self.options['num_gf'], 2),
			# Dim: batch_size x (num_gf) x 128 x 128
			nn.Conv2d(self.options['num_gf'], self.options['num_gf'], kernel_size=3, stride=1, padding=1),
			nn.ELU(inplace=True),
			# Dim: batch_size x (num_gf) x 128 x 128
			nn.Conv2d(self.options['num_gf'], self.options['num_gf'], kernel_size=3, stride=1, padding=1),
			nn.ELU(inplace=True),
			# Dim: batch_size x (num_gf) x 128 x 128
			nn.Conv2d(self.options['num_gf'], self.options['image_channels'], kernel_size=3, stride=1, padding=1),
			# Dim: batch_size x (num_image_channels) x 128 x 128
			nn.Tanh()
		)

		if self.options['verbose']: print('BEGAN Discriminator Output Created')
		if self.options['verbose']: print('BEGAN Discriminator Created\n')


	# BEGAN Discriminator Forward Propagation
	def forward(self, images, text_embed):
		images_intermediate = self.discriminator_input(images)
		projected_embed = self.d_projector(text_embed)
		# Repeat the projected dimensions and change the permutations
		# Dim: batch_size x 256 -> batch_size x 256 x 4 x 4
		replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
		latent_vec = torch.cat([images_intermediate, replicated_embed], 1)
		output = self.discriminator_output(latent_vec)

		return output


	# BEGAN Discriminator Loss
	# L_D = L(y_r) - k * L(y_f)
	# k = k + lambda_k * (gamma * L(y_r) + L(y_f))
	# BEGAN Discrminator Loss with CLS
	# L(y_w) is the caption loss sensitivity CLS (makes sure that captions match the image)
	# L_D = L(y_r) - k * (L(y_w) + L(y_f))
	# L_G = L(y_f)
	# k = k + lambda_k * (gamma * L(y_r) + L(y_w) + L(y_f))
	def loss(self, real_img, real_img_recons, fake_img, fake_img_recons, wrong_img=None, wrong_img_recons=None):
		if self.options['use_cls']:
			d_real_loss = torch.mean(torch.abs(real_img_recons - real_img))
			d_wrong_loss = torch.mean(torch.abs(wrong_img_recons - wrong_img))
			d_fake_loss = torch.mean(torch.abs(fake_img_recons - fake_img))

			d_loss = d_real_loss - self.began_k * (d_wrong_loss + d_fake_loss)

			# Update began k value
			balance = (self.options['began_gamma'] * d_real_loss + d_wrong_loss + d_fake_loss).data[0]
			self.began_k = min(max(self.began_k + self.options['began_lambda_k'] * balance, 0), 1)

		# No CLS option
		else:
			d_real_loss = torch.mean(torch.abs(real_img_recons - real_img))
			d_fake_loss = torch.mean(torch.abs(fake_img_recons - fake_img))

			d_loss = d_real_loss - self.began_k * d_fake_loss

			# Update began k value
			balance = (self.options['began_gamma'] * d_real_loss + d_fake_loss).data[0]
			self.began_k = min(max(self.began_k + self.options['began_lambda_k'] * balance, 0), 1)

		return d_loss

	# Calculates the gradients and returns the loss
	def calc_grad_d(self, real_img, real_img_recons, fake_img, fake_img_recons, wrong_img=None, wrong_img_recons=None):
		if self.options['use_cls']:
			d_loss = self.loss(real_img, real_img_recons, fake_img, fake_img_recons, wrong_img, wrong_img_recons)
		else:
			d_loss = self.loss(real_img, real_img_recons, fake_img, fake_img_recons)

		d_loss.backward()

		return d_loss
