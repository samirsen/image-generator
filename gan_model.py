'''
gan.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as f

class GAN:
    def __init__(self, options):
        self.options = options

        # Create the layers dict
        self.layers = {}

        # Add text input layers to the layers dict
        self.layers['text_embed_fc_layer'] = nn.Linear(self.options['caption_vec_len'], self.options['t_dim'])
        conv_vec_length = self.options['g_channels'] * gan_layer_filter_sizes[0]**2 * gan_layer_num_channels[0]
        self.layers['text_noise_fc_layer'] = nn.Linear(self.options['caption_vec_len'] + self.options['z_dim'], conv_vec_length)

        # Add hidden GAN layers to the layers dict
        for i in range(1, self.options['gan_num_layers'] + 1):
            self.layers['g_layer_' + i] = nn.ConvTranspose2d(self.options['gan_layer_num_channels'][i-1],                   \
                                                                self.options['gan_layer_num_channels'][i]),                 \
                                                                kernel_size = (self.options['gan_layer_filter_sizes'][i-1]  \
                                                                                self.options['gan_layer_filter_sizes'][i]))

    # Builds the GAN given instance, the text embeddings and the noise for the text embeddings
    def build_model(self, text_embed, noise):
        image_size = self.options['image_size']
        batch_size = self.options['batch_size']

        # Define data set
        # Make vector of text embeddings into a torch tensor
        t_text_embed = torch.Tensor(text_embed)     # dim: batch_size x caption_vec_len
        t_noise = torch.Tensor(noise)               # dim: batch_size x z_dim

        # Make fake image from generator
        fake_image = self.generator(t_text_embed, t_noise)

        # Run (real image, real caption), (wrong image, real caption), and (fake image, real caption) in discriminator

        # Calculate loss for both generator and discriminator



    #
    def generator(self, t_text_embed, t_noise):
        image_size = self.options['image_size']

        # Make text embeddings

        reduced_text_embed = f.leaky_relu(self.text_embed_fc_layer(t_text_embed), negative_slope=self.options['leak'])

        # Concatenate the noise and the reduced text embedding
        text_concat = torch.cat([t_noise, reduced_text_embed], dim=1)

        # Turn the noise and text concatenated tensor into a tensor that will be used for GAN input
        text_noise = self.text_noise_fc_layer(text_concat)

        # Create the text input tensors
        X = text_noise.view([-1, gan_layer_filter_sizes[0], gan_layer_filter_sizes[0], gan_layer_num_channels[0]])
        X = f.relu(text_input)

        # Go through each hidden layer of the GAN
        for i in range(1, self.options['gan_num_layers'] + 1):
            '''
            TODO ADD BATCH NORM
            '''
            X = self.layers['g_layer_' + i](X)
            if self.options['gan_layer_activation_func'] == 'relu':
                X = f.relu(X)
            elif self.options['gan_layer_activation_func'] == 'tanh'
                X = f.tanh(X)


        return X / 2. + 0.5

    def discriminator(self):
        image_size = self.options['image_size']

        '''
        TODO IMPLEMENT DISCRIMINATOR
        '''
