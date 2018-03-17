'''
train_dcgan.py
'''
import torch
import torch.optim as optim
import torch.nn.functional as func
from torch.autograd import Variable
from torch.utils.data import DataLoader
import constants
from model import *
from text_model import TextModel, LSTM_Model
import util
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip_longest
import matplotlib.pyplot as plt
import argparse
import time
import os
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
args = parser.parse_args()


def main():
    # Make directories if they don't already exist
    util.make_directories()
    # Load model options
    model_options = constants.MAIN_MODEL_OPTIONS

    ########## DATA ##########
    if constants.PRINT_MODEL_STATUS: print("Loading data")

    dataset_map = util.load_dataset_map()
    train_captions, val_captions, test_captions = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME, dataset_map)
    train_image_dict, val_image_dict, test_image_dict = util.get_images('Data', constants.DIRECTORY_PATH, constants.FLOWERS_DICTS_PATH)


    ########## MODEL ##########
    generator = Generator(model_options)
    discriminator = Discriminator(model_options)

    # Put G and D on cuda if GPU available
    if torch.cuda.is_available():
        if constants.PRINT_MODEL_STATUS: print("CUDA is available")
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if constants.PRINT_MODEL_STATUS: print("Moved models to GPU")

    # Initialize weights
    generator.apply(util.weights_init)
    discriminator.apply(util.weights_init)


    ########## SAVED VARIABLES #########
    new_epoch = 0
    train_losses = {"generator": [], "discriminator": []}
    val_losses = {"generator": [], "discriminator": []}
    losses = {'train': train_losses, 'val': val_losses}


    ########## OPTIMIZER ##########
    g_optimizer = optim.Adam(generator.parameters(), lr=constants.LR, betas=constants.BETAS)
    # Changes the optimizer to SGD if declared in constants
    if constants.D_OPTIMIZER_SGD: d_optimizer = optim.SGD(discriminator.parameters(), lr=constants.LR)
    else: d_optimizer = optim.Adam(discriminator.parameters(), lr=constants.LR, betas=constants.BETAS)
    if constants.PRINT_MODEL_STATUS: print("Added optimizers")


    ########## RESUME OPTION ##########
    if args.resume:
        print("Resuming from epoch " + args.resume)
        checkpoint = torch.load(constants.SAVE_PATH + 'weights/epoch' + str(args.resume))
        new_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['g_dict'])
        discriminator.load_state_dict(checkpoint['d_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        losses = torch.load(constants.SAVE_PATH + 'losses')


    ########## VARIABLES ##########
    noise_vec = torch.FloatTensor(constants.BATCH_SIZE, model_options['z_dim'], 1, 1)
    text_vec = torch.FloatTensor(constants.BATCH_SIZE, model_options['caption_vec_len'])
    real_img = torch.FloatTensor(constants.BATCH_SIZE, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
    real_caption = torch.FloatTensor(constants.BATCH_SIZE, model_options['caption_vec_len'])
    if constants.USE_CLS:
        wrong_img = torch.FloatTensor(constants.BATCH_SIZE, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
        wrong_caption = torch.FloatTensor(constants.BATCH_SIZE, model_options['caption_vec_len'])

    # Add cuda GPU option
    if torch.cuda.is_available():
        noise_vec = noise_vec.cuda()
        text_vec = text_vec.cuda()
        real_img = real_img.cuda()
        real_caption = real_caption.cuda()
        if constants.USE_CLS: wrong_img = wrong_img.cuda()


    ########## Training ##########
    num_iterations = 0
    for epoch in range(new_epoch, constants.NUM_EPOCHS):
        print("Epoch %d" % (epoch))
        st = time.time()

        for i, batch_iter in enumerate(util.grouper(train_captions.keys(), constants.BATCH_SIZE)):
            batch_keys = [x for x in batch_iter if x is not None]
            curr_batch_size = len(batch_keys)

            discriminator.train()
            generator.train()
            discriminator.zero_grad() # Zero out gradient
            # Save computations for gradient calculations
            for p in discriminator.parameters():
                p.requires_grad = True # Need this to be true to update generator as well


            ########## BATCH DATA #########
            noise_batch = torch.randn(curr_batch_size, model_options['z_dim'], 1, 1)
            text_vec_batch = torch.Tensor(util.get_text_description(train_captions, batch_keys))
            real_caption_batch = torch.Tensor(util.get_text_description(train_captions, batch_keys))
            real_img_batch = torch.Tensor(util.choose_real_image(train_image_dict, batch_keys))
            if constants.USE_CLS: wrong_img_batch = torch.Tensor(util.choose_wrong_image(train_image_dict, batch_keys))
            if torch.cuda.is_available():
                noise_batch = noise_batch.cuda()
                text_vec_batch = text_vec_batch.cuda()
                real_caption_batch = real_caption_batch.cuda()
                real_img_batch = real_img_batch.cuda()
                if constants.USE_CLS: wrong_img_batch = wrong_img_batch.cuda()

            # Fill in tensors with batch data
            noise_vec.resize_as_(noise_batch).copy_(noise_batch)
            text_vec.resize_as_(text_vec_batch).copy_(text_vec_batch)
            real_caption.resize_as_(text_vec_batch).copy_(text_vec_batch)
            real_img.resize_as_(real_img_batch).copy_(real_img_batch)
            if constants.USE_CLS: wrong_img.resize_as_(wrong_img_batch).copy_(wrong_img_batch)


            ########## RUN THROUGH GAN ##########
            gen_image = generator.forward(Variable(text_vec), Variable(noise_vec))

            real_img_passed = discriminator.forward(Variable(real_img), Variable(real_caption))
            fake_img_passed = discriminator.forward(gen_image.detach(), Variable(real_caption))
            if constants.USE_CLS: wrong_img_passed = discriminator.forward(Variable(wrong_img), Variable(real_caption))


            ########## TRAIN DISCRIMINATOR ##########
            # Overall loss function for discriminator
            # L_D = log(y_r) + log(1 - y_f)
            # Loss of Vanilla GAN with CLS
            # log(1 - y_w) is the caption loss sensitivity CLS (makes sure that captions match the image)
            # L_D = log(y_r) + log(1 - y_w) + log(1 - y_f)
            # Add one-sided label smoothing to the real images of the discriminator
            d_real_loss = func.binary_cross_entropy(real_img_passed, torch.ones_like(real_img_passed) - model_options['label_smooth'])
            d_fake_loss = func.binary_cross_entropy(fake_img_passed, torch.zeros_like(fake_img_passed))
            d_loss = d_real_loss + d_fake_loss

            # option to use conditional loss sensitivity
            if constants.USE_CLS:
                d_wrong_loss = func.binary_cross_entropy(wrong_img_passed, torch.zeros_like(wrong_img_passed))
                d_loss += d_wrong_loss

            d_loss.backward()
            d_optimizer.step()


            ########## TRAIN GENERATOR ##########
            generator.zero_grad()
            for p in discriminator.parameters():
                p.requires_grad = False

            # Generate image again if you want to
            if constants.REGEN_IMAGE:
                noise_batch = torch.randn(curr_batch_size, model_options['z_dim'], 1, 1)
                if torch.cuda.is_available():
                    noise_batch = noise_batch.cuda()
                noise_vec.resize_as_(noise_batch).copy_(noise_batch)
                gen_image = generator.forward(Variable(text_vec), Variable(noise_vec))

            new_fake_img_passed = discriminator.forward(gen_image, Variable(real_caption))

            # Generator Loss
            # L_G = log(y_f)
            g_loss = func.binary_cross_entropy(new_fake_img_passed, torch.ones_like(fake_img_passed))

            g_loss.backward()
            g_optimizer.step()

            # learning rate decay
            # g_optimizer = util.adjust_learning_rate(g_optimizer, num_iterations)
            # d_optimizer = util.adjust_learning_rate(d_optimizer, num_iterations)

            if i % constants.LOSS_SAVE_IDX == 0:
                losses['train']['generator'].append((g_loss.data[0], epoch, i))
                losses['train']['discriminator'].append((d_loss.data[0], epoch, i))

            num_iterations += 1

        print ('Total number of iterations: ', num_iterations)
        print ('Training G Loss: ', g_loss.data[0])
        print ('Training D Loss: ', d_loss.data[0])
        epoch_time = time.time()-st
        print ("Time: ", epoch_time)

        if epoch == constants.REPORT_EPOCH:
            with open(constants.SAVE_PATH + 'report.txt', 'w') as f:
                f.write(constants.EXP_REPORT)
                f.write("Time per epoch: " + str(epoch_time))
            print("Saved report")



        ########## DEV SET #########
        # Calculate dev set loss
        # Volatile is true because we are running in inference mode (no need to calculate gradients)
        generator.eval()
        discriminator.eval()
        for i, batch_iter in enumerate(util.grouper(val_captions.keys(), constants.BATCH_SIZE)):
            batch_keys = [x for x in batch_iter if x is not None]
            curr_batch_size = len(batch_keys)

            # Gather batch data
            noise_batch = torch.randn(curr_batch_size, model_options['z_dim'], 1, 1)
            text_vec_batch = torch.Tensor(util.get_text_description(val_captions, batch_keys))
            real_caption_batch = torch.Tensor(util.get_text_description(val_captions, batch_keys))
            real_img_batch = torch.Tensor(util.choose_real_image(val_image_dict, batch_keys))
            if constants.USE_CLS:
                wrong_img_batch = torch.Tensor(util.choose_wrong_image(val_image_dict, batch_keys))
            if torch.cuda.is_available():
                noise_batch = noise_batch.cuda()
                text_vec_batch = text_vec_batch.cuda()
                real_caption_batch = real_caption_batch.cuda()
                real_img_batch = real_img_batch.cuda()
                if constants.USE_CLS:
                    wrong_img_batch = wrong_img_batch.cuda()

            # Fill in tensors with batch data
            noise_vec.resize_as_(noise_batch).copy_(noise_batch)
            text_vec.resize_as_(text_vec_batch).copy_(text_vec_batch)
            real_caption.resize_as_(text_vec_batch).copy_(text_vec_batch)
            real_img.resize_as_(real_img_batch).copy_(real_img_batch)
            if constants.USE_CLS:
                wrong_img.resize_as_(wrong_img_batch).copy_(wrong_img_batch)


            # Run through generator
            gen_image = generator.forward(Variable(text_vec, volatile=True), Variable(noise_vec, volatile=True))   # Returns tensor variable holding image

            # Run through discriminator
            real_img_passed = discriminator.forward(Variable(real_img, volatile=True), Variable(real_caption, volatile=True))
            fake_img_passed = discriminator.forward(gen_image.detach(), Variable(real_caption, volatile=True))
            if constants.USE_CLS: wrong_img_passed = discriminator.forward(Variable(wrong_img, volatile=True), Variable(real_caption, volatile=True))


            # Calculate D loss
            d_real_loss = func.binary_cross_entropy(real_img_passed, torch.ones_like(real_img_passed) - model_options['label_smooth'])
            d_fake_loss = func.binary_cross_entropy(fake_img_passed, torch.zeros_like(fake_img_passed))
            d_loss = d_real_loss + d_fake_loss

            # option to use conditional loss sensitivity
            if constants.USE_CLS:
                d_wrong_loss = func.binary_cross_entropy(wrong_img_passed, torch.zeros_like(wrong_img_passed))
                d_loss += d_wrong_loss


            # Calculate G loss
            g_loss = func.binary_cross_entropy(fake_img_passed, torch.ones_like(fake_img_passed))

            if i % constants.LOSS_SAVE_IDX == 0:
                losses['val']['generator'].append((g_loss.data[0], epoch, i))
                losses['val']['discriminator'].append((d_loss.data[0], epoch, i))

        print ('Val G Loss: ', g_loss.data[0])
        print ('Val D Loss: ', d_loss.data[0])

        # Save losses
        torch.save(losses, constants.SAVE_PATH + 'losses')

        # Save images
        vutils.save_image(gen_image[0].data.cpu(),
                    constants.SAVE_PATH + 'images/gen0_epoch' + str(epoch) + '.png',
                    normalize=True)
        vutils.save_image(gen_image[1].data.cpu(),
                    constants.SAVE_PATH + 'images/gen1_epoch' + str(epoch) + '.png',
                    normalize=True)

        # Save model
        if epoch % 20 == 0 or epoch == constants.NUM_EPOCHS - 1:
            save_checkpoint = {
                'epoch': epoch,
                'g_dict': generator.state_dict(),
                'd_dict': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
            }
            torch.save(save_checkpoint, constants.SAVE_PATH + 'weights/epoch' + str(epoch))


if __name__ == '__main__':
    main()
