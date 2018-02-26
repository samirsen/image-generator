'''
gan.py
Based on the model from https://github.com/paarthneekhara/text-to-image
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import constants
import collections

'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
'''
class GAN(nn.Module):
    def __init__(self, options):
        self.options = options
        self.layers = collections.OrderedDict({})
        self.batch_norm = collections.OrderedDict({})

        if constants.PRINT_MODEL_STATUS: print('Creating Model...\n')

        # Add text input layers to the layers dict
        self.layers['g_text_embed_fc_layer'] = nn.Linear(self.options['caption_vec_len'], self.options['t_dim'])
        torch.nn.init.xavier_uniform(self.layers['g_text_embed_fc_layer'].weight)
        if constants.PRINT_MODEL_STATUS: print('Text Embedded Fully Connected Layer Created')

        # Calculate the length of the input of the GAN
        conv_vec_length = 1
        for i in self.options['g_layer_conv_sizes'][0][1:]:
            conv_vec_length *= i

        self.layers['g_text_noise_fc_layer'] = nn.Linear(self.options['t_dim'] + self.options['z_dim'], conv_vec_length)
        torch.nn.init.xavier_uniform(self.layers['g_text_noise_fc_layer'].weight)
        if constants.PRINT_MODEL_STATUS: print('Text Noise Fully Connected Layer Created')

        # Add hidden GAN generator layers to the layers dict
        for i in range(1, self.options['g_num_layers'] + 1):
            input_channels = self.options['g_layer_num_channels'][i-1]
            output_channels = self.options['g_layer_num_channels'][i]
            layer_filter_size = (self.options['g_layer_filter_sizes'][i],self.options['g_layer_filter_sizes'][i])
            layer_stride = self.options['g_layer_stride'][i]
            layer_padding = self.options['g_layer_padding'][i]

            # Create batch norm for each layer
            self.batch_norm['g_bn_layer_' + str(i)] = nn.BatchNorm2d(input_channels, self.options['bn_eps'], self.options['bn_momentum'])

            # Create conv transpose generator layer
            self.layers['g_layer_' + str(i)] = nn.ConvTranspose2d(input_channels, output_channels,      \
                                                                    kernel_size=layer_filter_size,      \
                                                                    stride=layer_stride,                \
                                                                    padding=layer_padding)

            torch.nn.init.xavier_uniform(self.layers['g_layer_' + str(i)].weight)
            if constants.PRINT_MODEL_STATUS: print('Generator Layer ' + str(i) + ' Created')

        # Added discriminator layers to layers dict
        for i in range(1, self.options['d_num_layers'] + 1):
            input_channels = self.options['d_layer_num_channels'][i-1]
            output_channels = self.options['d_layer_num_channels'][i]
            layer_filter_size = (self.options['d_layer_filter_sizes'][i],self.options['g_layer_filter_sizes'][i])
            layer_stride = self.options['d_layer_stride'][i]
            layer_padding = self.options['d_layer_padding'][i]

            # Create batch norm for each layer
            self.batch_norm['d_bn_layer_' + str(i)] = nn.BatchNorm2d(input_channels, self.options['bn_eps'], self.options['bn_momentum'])

            # Create conv layer
            self.layers['d_layer_' + str(i)] = nn.Conv2d(input_channels, output_channels,   \
                                                            kernel_size=layer_filter_size,  \
                                                            stride=layer_stride,            \
                                                            padding=layer_padding)
            torch.nn.init.xavier_uniform(self.layers['d_layer_' + str(i)].weight)
            if constants.PRINT_MODEL_STATUS: print('Discriminator Layer ' + str(i) + ' Created')

        # The discriminator text embedding fully connected layer for the reduced text embeddings
        self.layers['d_red_embed_fc_layer'] = nn.Linear(self.options['caption_vec_len'], self.options['t_dim'])
        torch.nn.init.xavier_uniform(self.layers['d_red_embed_fc_layer'].weight)

        # The conv layer of the concatenated images outputs from convolutional part of discriminator and the text embeddings
        cat_input_channels = self.options['d_layer_num_channels'][self.options['d_num_layers']] + self.options['t_dim']
        cat_output_channels = self.options['d_layer_num_channels'][self.options['d_num_layers']]
        self.layers['d_image_embed_conv_layer'] = nn.Conv2d(cat_input_channels, cat_output_channels,    \
                                                        kernel_size=(1,1),                      \
                                                        stride=(1,1),                           \
                                                        padding=0)
        torch.nn.init.xavier_uniform(self.layers['d_image_embed_conv_layer'].weight)

        # The fully connected layer of the output
        fc_dim = 1
        for i in self.options['d_layer_conv_sizes'][self.options['d_num_layers']][1:]:
            fc_dim *= i
        self.layers['d_output_fc_layer'] = nn.Linear(fc_dim, 1)

        print('Entire Model Created\n')


    # Takes in the instance, the text embeddings (batch_size x caption_vec_len), and the noise vector (batch_size x z_dim)
    # Generates the fake images
    def generate(self, text_embed, noise):
        image_size = self.options['image_size']

        # Make vector of text embeddings into a torch tensor and then variable
        t_text_embed = Variable(torch.Tensor(text_embed))     # dim: batch_size x caption_vec_len
        t_noise = Variable(torch.Tensor(noise))               # dim: batch_size x z_dim

        # Make text embeddings
        reduced_text_embed = self.layers['g_text_embed_fc_layer'](t_text_embed)
        reduced_text_embed = f.leaky_relu(reduced_text_embed, negative_slope=self.options['leak'])

        print t_noise.shape, reduced_text_embed.shape
        # Concatenate the noise and the reduced text embedding
        text_concat = torch.cat([t_noise, reduced_text_embed], dim=1)

        # Turn the noise and text concatenated tensor into a tensor that will be used for GAN input
        text_noise = self.layers['g_text_noise_fc_layer'](text_concat)

        # Create the text input tensors
        X = text_noise.view(self.options['g_layer_conv_sizes'][0])
        X = f.relu(X)

        # Go through each hidden layer of the generator
        for i in range(1, self.options['g_num_layers'] + 1):
            # Apply batch norm
            X = self.batch_norm['g_bn_layer_' + str(i)](X)

            # Run conv transpose layer
            new_size = self.options['g_layer_conv_sizes'][i]
            X = self.layers['g_layer_' + str(i)](X, output_size=new_size)

            # Run activation functions
            if self.options['g_layer_activation_func'] == 'relu':
                X = f.relu(X)
            elif self.options['g_layer_activation_func'] == 'tanh':
                X = f.tanh(X)

        # return X / 2. + 0.5
        return X

    # Takes in the variable versions of the tensors image_vec (BATCH_SIZE, CHANNELS, H, W) and text_embed (batch_size x caption_vec_len)
    #
    def discriminate(self, t_image_vec, t_text_embed):
        image_size = self.options['image_size']
        X = t_image_vec

        # Go through each hidden layer of the discriminator
        for i in range(1, self.options['d_num_layers'] + 1):
            # Apply batch norm
            X = self.batch_norm['d_bn_layer_' + str(i)](X)

            # Run conv layer
            X = self.layers['d_layer_' + str(i)](X)

            if self.options['d_layer_activation_func'][i] == 'lrelu':
                X = f.relu(X)

        # Add text embedding
        red_text_embed = self.layers['d_red_embed_fc_layer'](t_text_embed)
        red_text_embed = f.leaky_relu(red_text_embed, negative_slope=self.options['leak'])

        # Expand dimensions
        red_text_embed = red_text_embed.unsqueeze(1)
        red_text_embed = red_text_embed.unsqueeze(2)
        repeat_embed = red_text_embed.repeat(1, constants.D_EMBED_EXPAND, constants.D_EMBED_EXPAND, 1)
        # NEED TO SWITCH DIMENSIONS
        # (BATCHSIZE, HEIGHT, WIDTH, CHANNELS) -> (BATCHSIZE, CHANNELS, HEIGHT, WIDTH)
        repeat_embed = repeat_embed.permute(0,3,1,2)

        # Concatenate the output of the image convolution and the text embeddings
        X_concat = torch.cat([X, repeat_embed], dim=1)
        X_concat = self.layers['d_image_embed_conv_layer'](X_concat)
        X_concat = f.leaky_relu(X_concat, negative_slope=self.options['leak'])

        # The fully connected layer of both the image convolution and the text embeddings
        X_fc = X_concat.view(X_concat.shape[0], -1)
        X_fc = self.layers['d_output_fc_layer'](X_fc)

        return f.sigmoid(X_fc)
