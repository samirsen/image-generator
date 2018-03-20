import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import constants
from model import Generator, Discriminator, BeganGenerator, BeganDiscriminator
from lstm_model import LSTM
from vocab import get_glove
from util import *
from captions_utils import *
from train_utils import *
from data_batcher import *
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--resume')
args = parser.parse_args()

def load_glove(paths):
    embeddings, word2id, id2word = (torch.load(path) for path in paths)
    return embeddings, word2id, id2word

def _init_lstm(m):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal(param)

def lstm_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal(param)

def main():
    print("Starting LSTM training for CLS GAN ...")

    ########## SAVED VARIABLES #########
    new_epoch = 0
    train_losses = {"generator": [], "discriminator": []}
    val_losses = {"generator": [], "discriminator": []}
    losses = {'train': train_losses, 'val': val_losses}

    model_options = constants.MAIN_MODEL_OPTIONS
    caption_dict = load_flowers_capt_dict(data_dir='Data')   # filename --> [captions]
    img_dict = load_image_dict()   # filename --> 28 x 28 image

    if os.path.exists('Data/vocab/glove_matrix.torch'):
        paths = ['Data/vocab/glove_matrix.torch', 'Data/vocab/word_to_idx.torch', 'Data/vocab/idx_to_word.torch']
        embeddings, word2id, id2word = load_glove(paths)
    else:
        emb_matrix, word2id, id2word = get_glove(constants.GLOVE_PATH, constants.EMBED_DIM)
        embeddings = torch.from_numpy(emb_matrix).float()
        torch.save(embeddings, 'Data/vocab/glove_matrix.torch')
        torch.save(word2id, 'Data/vocab/word_to_idx.torch')
        torch.save(id2word, 'Data/vocab/idx_to_word.torch')

    print ( "shape of embedding size: ", embeddings.size() )

    lstm = LSTM(model_options, embeddings)
    lstm_weights(lstm)

    if torch.cuda.is_available(): lstm = lstm.cuda()

    generator, discriminator = choose_model(model_options)
    g_optimizer, d_optimizer = choose_optimizer(generator, discriminator)
    lstm_optimizer = optim.Adam(lstm.parameters(), lr=constants.LR, betas=constants.BETAS)

    ########## RESUME OPTION ##########
    if args.resume:
        print("Resuming from epoch " + args.resume)
        checkpoint = torch.load(constants.SAVE_PATH + 'weights/epoch' + str(args.resume))
        new_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['g_dict'])
        discriminator.load_state_dict(checkpoint['d_dict'])
        lstm.load_state_dict(checkpoint['lstm_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        losses = torch.load(constants.SAVE_PATH + 'losses')

    ########## VARIABLES ##########
    noise_vec = torch.FloatTensor(constants.BATCH_SIZE, model_options['z_dim'], 1, 1)
    # text_vec = torch.FloatTensor(constants.BATCH_SIZE, model_options['caption_vec_len'])
    real_img = torch.FloatTensor(constants.BATCH_SIZE, model_options['image_channels'], constants.IMAGE_SIZE, constants.IMAGE_SIZE)
    real_caption = torch.FloatTensor(constants.BATCH_SIZE, model_options['caption_vec_len'])
    if constants.USE_CLS:
        wrong_img = torch.FloatTensor(constants.BATCH_SIZE, model_options['image_channels'], constants.IMAGE_SIZE, constants.IMAGE_SIZE)
        wrong_caption = torch.FloatTensor(constants.BATCH_SIZE, model_options['caption_vec_len'])

    # Add cuda GPU option
    if torch.cuda.is_available():
        noise_vec = noise_vec.cuda()
        # text_vec = text_vec.cuda()
        real_img = real_img.cuda()
        # real_caption = real_caption.cuda()
        if constants.USE_CLS: wrong_img = wrong_img.cuda()

    ################################
    # Now get batch of captions and glove embeddings
    # Use this batch as input to BiRNN w LSTM cells
    # Use generator loss to update lstm -- look into line 229, main.py
    # TODO: Loop over epochs in constants.NUM_EPOCHS
    ################################
    num_iterations = 0
    for epoch in range(constants.NUM_EPOCHS):
        print("Epoch %d" % (epoch))
        st = time.time()

        for i, batch_iter in enumerate(grouper(caption_dict.keys(), constants.BATCH_SIZE)):
            batch_keys = [x for x in batch_iter if x is not None]
            if len(batch_keys) < constants.BATCH_SIZE: continue
            curr_batch_size = len(batch_keys)

            init_model(discriminator, generator, lstm)

            ########## BATCH DATA #########
            noise_batch = torch.randn(curr_batch_size, model_options['z_dim'], 1, 1)
            caption_embeds, real_embeds = text_model(batch_keys, caption_dict, word2id, lstm)
            real_img_batch = torch.Tensor(choose_real_image(img_dict, batch_keys))
            if constants.USE_CLS: wrong_img_batch = torch.Tensor(util.choose_wrong_image(train_image_dict, batch_keys))
            if torch.cuda.is_available():
                noise_batch = noise_batch.cuda()
                real_img_batch = real_img_batch.cuda()
                if constants.USE_CLS: wrong_img_batch = wrong_img_batch.cuda()

            # Fill in tensors with batch data
            noise_vec.resize_as_(noise_batch).copy_(noise_batch)
            # text_vec.resize_as_(caption_embeds).copy_(caption_embeds)
            # real_caption.resize_as_(real_embeds).copy_(real_embeds)
            real_img.resize_as_(real_img_batch).copy_(real_img_batch)
            if constants.USE_CLS: wrong_img.resize_as_(wrong_img_batch).copy_(wrong_img_batch)



            # Returns variable tensor of size (BATCH_SIZE, 1, 4800)
            # caption_embeds, real_embeds = text_model(batch_keys, caption_dict, word2id, lstm)

            # real_img_batch = torch.Tensor(choose_real_image(img_dict, batch_keys))
            # wrong_img_batch = torch.Tensor(choose_wrong_image(img_dict, batch_keys))

            # Run through generator
            gen_image = generator.forward(Variable(text_vec), Variable(noise_vec))

            real_img_passed = discriminator.forward(Variable(real_img_batch), Variable(real_caption))
            fake_img_passed = discriminator.forward(gen_image.detach(), Variable(real_caption))
            wrong_img_passed = discriminator.forward(Variable(wrong_img_batch), Variable(real_caption))

            ########## TRAIN DISCRIMINATOR ##########
            # Overall loss function for discriminator
            # L_D = log(y_r) + log(1 - y_f)
            # Loss of Vanilla GAN with CLS
            # log(1 - y_w) is the caption loss sensitivity CLS (makes sure that captions match the image)
            # L_D = log(y_r) + log(1 - y_w) + log(1 - y_f)
            # Add one-sided label smoothing to the real images of the discriminator
            d_real_loss = f.binary_cross_entropy(real_img_passed, torch.ones_like(real_img_passed) - model_options['label_smooth'])
            d_fake_loss = f.binary_cross_entropy(fake_img_passed, torch.zeros_like(fake_img_passed))
            d_wrong_loss = f.binary_cross_entropy(wrong_img_passed, torch.zeros_like(wrong_img_passed))
            d_loss = d_real_loss + d_fake_loss + d_wrong_loss

            d_loss.backward()
            d_optimizer.step()

            ########## TRAIN GENERATOR ##########
            generator.zero_grad()
            for p in discriminator.parameters():
                p.requires_grad = False

            # Regenerate the image
            noise_vec = torch.randn(constants.BATCH_SIZE, model_options['z_dim'], 1, 1)
            if torch.cuda.is_available():noise_vec = noise_vec.cuda()
            gen_image = generator.forward(Variable(text_vec), Variable(noise_vec))

            new_fake_img_passed = discriminator.forward(gen_image, real_embeds)
            g_loss = f.binary_cross_entropy(new_fake_img_passed, torch.ones_like(fake_img_passed))

            g_loss.backward()
            g_optimizer.step()

            ########## TRAIN LSTM ##############
            lstm.zero_grad()
            lstm_loss = g_loss

            lstm_loss.backward()
            lstm_optimizer.step()

            if i % constants.LOSS_SAVE_IDX == 0:
                losses['train']['generator'].append((g_loss.data[0], epoch, i))
                losses['train']['discriminator'].append((d_loss.data[0], epoch, i))

            num_iterations += 1
            print ('batch ' + str(i) + ' complete.')

        print ('Total number of iterations: ', num_iterations)
        print ('Training G Loss: ', g_loss.data[0])
        print ('Training D Loss: ', d_loss.data[0])
        epoch_time = time.time()-st
        print ("Time: ", epoch_time)

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
        if epoch % constants.CHECKPOINT_FREQUENCY == 0 and epoch != 0 or epoch == constants.NUM_EPOCHS - 1:
            save_checkpoint = {
                'epoch': epoch,
                'g_dict': generator.state_dict(),
                'd_dict': discriminator.state_dict(),
                'lstm_dict': lstm.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
            }
            torch.save(save_checkpoint, constants.SAVE_PATH + 'weights/epoch' + str(epoch))


if __name__ == '__main__':
    main()
