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





class Generator(nn.Module):
	def __init__(self, options):
		super(Generator, self).__init__()

		self.options = options
		# Dimensions of the latent vector (concatenate processed embedding vector and noise vector)
		self.options['concat_dim'] = self.options['t_dim'] + self.options['z_dim']

		if constants.PRINT_MODEL_STATUS: print('\nCreating Generator...')

		# Projector processes the word embedding before we concatenate embedding with noise
		self.g_projector = nn.Sequential(
			nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if constants.PRINT_MODEL_STATUS: print('Generator Projector Created')

		# Generator inputs concated word embedding and noise vector (latent vector) and outputs image
		self.generator = nn.Sequential(
			# Input Dim: batch_size x (concat_dim) x 1 x 1
			nn.ConvTranspose2d(self.options['concat_dim'], self.options['num_gf'] * 16, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 16),
			nn.ReLU(inplace=True),
			# Dim: batch_size x (num_gf * 16) x 4 x 4
			nn.ConvTranspose2d(self.options['num_gf'] * 16, self.options['num_gf'] * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 8),
			nn.ReLU(inplace=True),
			# Dim: batch_size x (num_gf * 8) x 8 x 8
			nn.ConvTranspose2d(self.options['num_gf'] * 8, self.options['num_gf'] * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 4),
			nn.ReLU(inplace=True),
			# Dim: batch_size x (num_gf * 4) x 16 x 16
			nn.ConvTranspose2d(self.options['num_gf'] * 4, self.options['num_gf'] * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf'] * 2),
			nn.ReLU(inplace=True),
			# Dim: batch_size x (num_gf * 2) x 32 x 32
			nn.ConvTranspose2d(self.options['num_gf'] * 2, self.options['num_gf'], 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.options['num_gf']),
			nn.ReLU(inplace=True),
			# Dim: batch_size x (num_gf) x 64 x 64
			nn.ConvTranspose2d(self.options['num_gf'], self.options['image_channels'], 4, 2, 1, bias=False),
			nn.Tanh()
			# Dim: batch_size x (num_channels) x 128 x 128
		)

		if constants.PRINT_MODEL_STATUS: print('Generator Created\n')

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
	def loss(self, logits):
		g_loss = f.binary_cross_entropy(logits, torch.ones_like(logits))

		return g_loss


class Discriminator(nn.Module):
	def __init__(self, options):
		super(Discriminator, self).__init__()

		self.options = options

		if constants.PRINT_MODEL_STATUS: print('Creating Discriminator...')

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

		if constants.PRINT_MODEL_STATUS: print('Discriminator Input Created')

		# Discriminator layers for the projection of the text embedding
		self.d_projector = nn.Sequential(
		    nn.Linear(in_features=self.options['caption_vec_len'], out_features=self.options['t_dim']),
			nn.BatchNorm1d(num_features=self.options['t_dim']),
			nn.LeakyReLU(negative_slope=self.options['leak'], inplace=True)
		)

		if constants.PRINT_MODEL_STATUS: print('Discriminator Projector Created')

		# Discriminator layers for the concatenation of the text embedding and image
		self.discriminator_output = nn.Sequential(
			# state size. (num_df * 8) x 8 x 8
			nn.Conv2d(self.options['num_df'] * 16 + self.options['t_dim'], 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

		if constants.PRINT_MODEL_STATUS: print('Discriminator Output Created')
		if constants.PRINT_MODEL_STATUS: print('Discriminator Created\n')

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

	# Discriminator Loss
	# L_D = log(y_r) + log(1 - y_w) + log(1 - y_f)
	def loss(self, real_img_passed, wrong_img_passed, fake_img_passed):
		d_loss1 = f.binary_cross_entropy(real_img_passed, torch.ones_like(real_img_passed))
		d_loss2 = f.binary_cross_entropy(wrong_img_passed, torch.zeros_like(wrong_img_passed))
		d_loss3 = f.binary_cross_entropy(fake_img_passed, torch.zeros_like(fake_img_passed))

		d_loss = d_loss1 + d_loss2 + d_loss3

		return d_loss