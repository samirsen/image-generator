'''
main.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import constants
from model import Generator, Discriminator, BeganGenerator, BeganDiscriminator
import util
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip_longest
import scipy.misc
import matplotlib.pyplot as plt
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
args = parser.parse_args()

# From https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
# Iterates over an array in chunks
def grouper(array, n):
    args = [iter(array)] * n
    return izip_longest(*args)

# Show the generated image improves over time
def print_images(generated):
    for img in generated:
        image_done = img.data.numpy()
        swap_image = np.swapaxes(image_done,1,2)
        swap_image = np.swapaxes(swap_image,2,3)
        plt.imshow(swap_image[0])
        plt.show()

def get_text_description(text_caption_dict, batch_keys):
    g_idx = [np.random.randint(len(text_caption_dict[batch_keys[0]])) for i in range(len(batch_keys))]
    g_text_des = np.array([text_caption_dict[k][i] for k,i in zip(batch_keys, g_idx)])
    # g_text_des = np.expand_dims(g_text_des, axis=0) ONLY NEED FOR 1 DIM

    return g_text_des

def choose_wrong_image(image_dict, batch_keys):
    wrong_image = []
    for k in batch_keys:
        wrong_key = np.random.choice(image_dict.keys())
        while wrong_key == k:
            wrong_key = np.random.choice(image_dict.keys())

        wrong_image.append(image_dict[wrong_key])
    wrong_image = np.array(wrong_image)
    wrong_image = augment_image_batch(wrong_image)
    wrong_image = np.swapaxes(wrong_image, 2, 3)
    wrong_image = np.swapaxes(wrong_image, 1, 2)
    return wrong_image

# Finds the true image for the given batch data
def choose_true_image(image_dict, batch_keys):
    true_img = np.array([image_dict[k] for k in batch_keys])
    true_img = augment_image_batch(true_img)
    true_img = np.swapaxes(true_img, 2, 3)
    true_img = np.swapaxes(true_img, 1, 2)
    return true_img

def augment_image_batch(images):
    batch_size = images.shape[0]
    for i in range(batch_size):
        curr = images[i, :, :, :]
        if np.random.rand() > .5:
            curr = np.flip(curr, 1)
        images[i, :, :, :] = curr
    return images

def generate_step(text_caption_dict, noise_vec, batch_keys, generator):
    g_text_des = get_text_description(text_caption_dict, batch_keys)
    g_text_des = Variable(torch.Tensor(g_text_des))
    if torch.cuda.is_available():
        g_text_des = g_text_des.cuda()
    gen_image = generator.forward(g_text_des, noise_vec)   # Returns tensor variable holding image

    return gen_image



def main():
    print("Starting..")
    output_path = constants.SAVE_PATH
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Made output directory")
    else:
        print("WARNING: starting training with an existing outputs directory")
    if not os.path.exists(output_path + 'weights/'):
        os.makedirs(output_path + 'weights/')
        print("Made weights directory")
    if not os.path.exists(output_path + 'images/'):
        os.makedirs(output_path + 'images/')
        print("Made images directory")
    model_options = constants.MAIN_MODEL_OPTIONS
    # Load map mapping examples to their train/dev/test split
    dataset_map = util.load_dataset_map()
    print("Loading data")
    # Load the caption text vectors
    train_captions, val_captions, test_captions = util.load_text_vec('Data', constants.VEC_OUTPUT_FILE_NAME, dataset_map)

    # Loads and separates images into train, dev, and test sets
    if os.path.exists(constants.FLOWERS_DICTS_PATH):
        image_dicts = torch.load(constants.FLOWERS_DICTS_PATH)
        train_image_dict, val_image_dict, test_image_dict = image_dicts
        print("Loaded images")
    else:
        print("Loading images and separating into train/val/test sets")
        filenames = train_captions.keys() + val_captions.keys() + test_captions.keys()
        train_image_dict, val_image_dict, test_image_dict = util.load_images('Data/' + constants.DIRECTORY_PATH, filenames, dataset_map)
        image_dicts = [train_image_dict, val_image_dict, test_image_dict]
        torch.save(image_dicts, "Data/flowers_dicts.torch")



    # Creates the model (BEGAN vs GAN/WGAN)
    if constants.USE_BEGAN_MODEL:
        generator = BeganGenerator(model_options)
        discriminator = BeganDiscriminator(model_options)
    else:
        generator = Generator(model_options)
        discriminator = Discriminator(model_options)


    if torch.cuda.is_available():
        print("CUDA is available")
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        print("Moved models to GPU")

    # Initialize weights
    generator.apply(util.weights_init)
    discriminator.apply(util.weights_init)

    new_epoch = 0
    train_losses = {"generator": [], "discriminator": []}
    val_losses = {"generator": [], "discriminator": []}
    losses = {'train': train_losses, 'val': val_losses}
    if args.resume:
        print("Resuming from epoch " + args.resume)
        new_epoch = int(args.resume) + 1
        gen_state = torch.load(constants.SAVE_PATH + 'weights/g_epoch' + str(args.resume))
        generator.load_state_dict(gen_state)
        dis_state = torch.load(constants.SAVE_PATH + 'weights/d_epoch' + str(args.resume))
        discriminator.load_state_dict(dis_state)
        losses = torch.load(constants.SAVE_PATH + 'losses')

    g_optimizer = optim.Adam(generator.parameters(), lr=constants.LR, betas=constants.BETAS)
    # Changes the optimizer to SGD if declared in constants
    if constants.D_OPTIMIZER_SGD:
        d_optimizer = optim.SGD(discriminator.parameters(), lr=constants.LR)
    else:
        d_optimizer = optim.Adam(discriminator.parameters(), lr=constants.LR, betas=constants.BETAS)

    print("Added optimizers")


    # TODO: MAKE SURE IMAGES ARE OF DIMENSIONS (BATCHSIZE, CHANNELS, H, W)
    # TODO: ADD L1/L2 Regularizaiton
    # TODO: USE DATALOADER FROM TORCH UTILS!!!!!!!!!
    # data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    # TODO: ADD PARALLELIZATION
    # TODO: ADD IMAGE PREPROCESSING? DO WE NEED TO SUBTRACT/ADD ANYTHING TO IMAGES

    # TODO: Add image aug


    # Grad factor alters whether we have gradient descent (grad_factor = 1) or gradient ascent (grad_factor = -1)
    if constants.USE_WGAN_MODEL:
        # WGAN uses gradient ascent
        grad_factor = Variable(torch.Tensor([-1]))
    else:
        # Other models use gradient descent
        grad_factor = Variable(torch.Tensor([1]))

    if torch.cuda.is_available():
        grad_factor = grad_factor.cuda()

    # Loop over dataset N times
    for epoch in range(new_epoch, constants.NUM_EPOCHS):
        print("Epoch %d" % (epoch))
        st = time.time()
        for i, batch_iter in enumerate(grouper(train_captions.keys(), constants.BATCH_SIZE)):
            batch_keys = [x for x in batch_iter if x is not None]
            noise_vec = Variable(torch.randn(len(batch_keys), model_options['z_dim'], 1, 1))
            if torch.cuda.is_available():
                noise_vec = noise_vec.cuda()

            discriminator.train()
            generator.train()
            # Zero out gradient
            discriminator.zero_grad()

            # Get batch data
            true_caption = get_text_description(train_captions, batch_keys)
            true_img = choose_true_image(train_image_dict, batch_keys)
            wrong_img = choose_wrong_image(train_image_dict, batch_keys)

            # Run through generator
            gen_image = generate_step(train_captions, noise_vec, batch_keys, generator)

            # Run through discriminator
            if torch.cuda.is_available():
                real_img_passed = discriminator.forward(Variable(torch.Tensor(true_img)).cuda(), Variable(torch.Tensor(true_caption)).cuda())
                wrong_img_passed = discriminator.forward(Variable(torch.Tensor(wrong_img)).cuda(), Variable(torch.Tensor(true_caption)).cuda())
                fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)).cuda())
            else:
                real_img_passed = discriminator.forward(Variable(torch.Tensor(true_img)), Variable(torch.Tensor(true_caption)))
                wrong_img_passed = discriminator.forward(Variable(torch.Tensor(wrong_img)), Variable(torch.Tensor(true_caption)))
                fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)))

            # Train discriminator
            if constants.USE_BEGAN_MODEL:
                d_loss = discriminator.began_loss(real_img_passed, wrong_img_passed, fake_img_passed)
            else:
                d_loss = discriminator.loss(real_img_passed, wrong_img_passed, fake_img_passed)
            d_loss.backward(grad_factor, retain_graph=True) # Since backprop of generator uses same output graph, retain it
            d_optimizer.step()

            # Train generator
            generator.zero_grad()
            if torch.cuda.is_available():
                new_fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)).cuda())
            else:
                new_fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)))
            g_loss = generator.loss(new_fake_img_passed)
            g_loss.backward(grad_factor)
            g_optimizer.step()

            # Update k value for BEGAN model
            if constants.USE_BEGAN_MODEL:
                balance = constants.BEGAN_GAMMA * original_d_loss
                k = min(max(k + constants.LAMBDA_K * balance, 0), 1)

            if i % constants.LOSS_SAVE_IDX == 0:
                losses['train']['generator'].append((g_loss.data[0], epoch, i))
                losses['train']['discriminator'].append((d_loss.data[0], epoch, i))

        print ('Training G Loss: ', g_loss.data[0])
        print ('Training D Loss: ', d_loss.data[0])
        epoch_time = time.time()-st
        print ("Time: ", epoch_time)

        if epoch == constants.REPORT_EPOCH:
            with open(constants.SAVE_PATH + 'report.txt', 'w') as f:
                f.write(constants.EXP_REPORT)
                f.write("Time per epoch: " + str(epoch_time))
            print("Saved report")

        # Calculate dev set loss
        generator.eval()
        discriminator.eval()
        for i, batch_iter in enumerate(grouper(val_captions.keys(), constants.BATCH_SIZE)):
            batch_keys = [x for x in batch_iter if x is not None]
            noise_vec = Variable(torch.randn(len(batch_keys), model_options['z_dim'], 1, 1))
            if torch.cuda.is_available():
                noise_vec = noise_vec.cuda()

            # Get batch data
            true_caption = get_text_description(val_captions, batch_keys)
            true_img = choose_true_image(val_image_dict, batch_keys)
            wrong_img = choose_wrong_image(val_image_dict, batch_keys)

            # Run through generator
            gen_image = generate_step(val_captions, noise_vec, batch_keys, generator)

            # Run through discriminator
            if torch.cuda.is_available():
                real_img_passed = discriminator.forward(Variable(torch.Tensor(true_img)).cuda(), Variable(torch.Tensor(true_caption)).cuda())
                wrong_img_passed = discriminator.forward(Variable(torch.Tensor(wrong_img)).cuda(), Variable(torch.Tensor(true_caption)).cuda())
                fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)).cuda())
            else:
                real_img_passed = discriminator.forward(Variable(torch.Tensor(true_img)), Variable(torch.Tensor(true_caption)))
                wrong_img_passed = discriminator.forward(Variable(torch.Tensor(wrong_img)), Variable(torch.Tensor(true_caption)))
                fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)))

            d_loss = discriminator.loss(real_img_passed, wrong_img_passed, fake_img_passed)
            if torch.cuda.is_available():
                new_fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)).cuda())
            else:
                new_fake_img_passed = discriminator.forward(gen_image, Variable(torch.Tensor(true_caption)))
            g_loss = generator.loss(new_fake_img_passed)

            if i % constants.LOSS_SAVE_IDX == 0:
                losses['val']['generator'].append((g_loss.data[0], epoch, i))
                losses['val']['discriminator'].append((d_loss.data[0], epoch, i))

        print ('Val G Loss: ', g_loss.data[0])
        print ('Val D Loss: ', d_loss.data[0])



        # Save losses
        torch.save(losses, constants.SAVE_PATH + 'losses')

        # Save images
        currImage = gen_image[0].data.cpu()
        currImage = currImage.numpy()
        currImage = np.swapaxes(currImage, 0, 1)
        currImage = np.swapaxes(currImage, 1, 2)
        scipy.misc.imsave(constants.SAVE_PATH + 'images/epoch' + str(epoch) + '.png', currImage)
        # Save model
        if epoch % 20 == 0 or epoch == constants.NUM_EPOCHS - 1:
            torch.save(generator.state_dict(), constants.SAVE_PATH + 'weights/g_epoch' + str(epoch))
            torch.save(discriminator.state_dict(), constants.SAVE_PATH + 'weights/d_epoch' + str(epoch))



    # FOR TESTING
    # for k in text_caption_dict:
    #     noise_vec = torch.randn(5, model_options['z_dim'], 1, 1)
    #     image = generator.forward(Variable(torch.Tensor(text_caption_dict[k])), Variable(torch.Tensor(noise_vec)))
    #     output = discriminator.forward(image, Variable(torch.Tensor(text_caption_dict[k])))
    #     print "DISCRIM OUTPUT", output
    #     break
    # print image.shape
    # swap_image = image.data.numpy()[0]
    # swap_image = np.swapaxes(swap_image,0,1)
    # swap_image = np.swapaxes(swap_image,1,2)
    # print swap_image.shape
    # plt.imshow(swap_image)
    # plt.show()
    # END TESTING

    # TESTING Discriminator
    # PYTORCH HAS DIMENSIONS (BATCHSIZE, CHANNELS, H, W)
    # NEED TO SWITCH FROM (BATCHSIZE, H, W, CHANNELS)
    # for i in image_dict:
    #     image_dict[i] = np.swapaxes(image_dict[i],1,2)
    #     image_dict[i] = np.swapaxes(image_dict[i],0,1)
    #     image_dict[i] = np.expand_dims(image_dict[i], axis=0)
    #     text_des = text_caption_dict[i][0]
    #     text_des = np.expand_dims(text_des, 0)
    #     output = gan.discriminate(Variable(torch.Tensor(image_dict[i])), Variable(torch.Tensor(text_des)))

    # print output

if __name__ == '__main__':
    main()
