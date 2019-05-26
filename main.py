#!/usr/bin/python
import argparse
import numpy as np
import sys

import torch
from torchvision.utils import save_image

from classifier import Classifier
from discriminator import Discriminator
from generator import Generator
from model import GAN_Model

from loader import get_data_loader
from score_fid import get_sample_loader
from score_fid import get_fid_score

from utils import create_directories
from utils import get_sample_batch
from utils import generate_samples


def train(G, D, args, trainloader):
    
    D_total = 0
    G_total = 0
    
    # loss for class predictions
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        
        ########## DISCRIMINATOR LOSS ##########
        D.optimizer.zero_grad()
        
        # retrieve real images
        real_images = data.to(device)
        real_labels = labels.to(device)
        
        # generate fake images
        samples, fake_labels = get_sample_batch(data.size(0), args.z_size, args.n_classes)
        fake_images = G(samples)
        
        # wasserstein loss
        real_scores, real_logits = D(real_images)
        fake_scores, fake_logits = D(fake_images)
        real_loss = torch.mean(real_scores)
        fake_loss = torch.mean(fake_scores)
        wgan_loss = fake_loss - real_loss
        
        # gradient penalty
        alpha = torch.rand([data.size(0), 1, 1, 1]).to(device)
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images
        interpolated_scores, _ = D(interpolated_images)
        interpolated_grad = torch.autograd.grad(
            outputs = interpolated_scores,
            inputs = interpolated_images,
            grad_outputs = torch.ones([data.size(0), 1]).to(device),
            create_graph = True,
            retain_graph = True,
            only_inputs = True)[0]
        grad_norm = torch.norm(interpolated_grad, 2, dim=1)
        grad_penalty = 10 * torch.mean((grad_norm - 1)**2)
        
        # class prediction loss
        real_class_loss = criterion(real_logits, real_labels)
        #fake_class_loss = criterion(fake_logits, fake_labels)
        class_loss_D = real_class_loss# + fake_class_loss
        
        # discriminator full loss and parameter update
        D_loss = wgan_loss + grad_penalty + 100*class_loss_D
        D_loss.backward()
        D.optimizer.step()
        
        D_total += D_loss * data.size(0)
        
        ############ GENERATOR LOSS ############
        if batch_idx % args.critics == 0:
            
            G.optimizer.zero_grad()
            
            # generate fake images
            samples, fake_labels = get_sample_batch(data.size(0), args.z_size, args.n_classes)
            fake_images = G(samples)
            
            # class prediction loss
            fake_scores, fake_logits = D(fake_images)
            class_loss_G = criterion(fake_logits, fake_labels)
            
            # generator full loss and parameter update
            G_loss = 100*class_loss_G - torch.mean(fake_scores)
            G_loss.backward()
            G.optimizer.step()
            
            G_total += G_loss * data.shape[0]
        
        # batch info display
        if (batch_idx + 1) % int(len(trainloader.dataset) / len(data) / 10) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD_loss: {:.6f}\tG_loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100 * batch_idx / len(trainloader),
                D_loss.item(), G_loss.item()))
    
    print('wgan_loss =', wgan_loss.item())
    print('====> Average epoch loss:\tD_loss: {:.4f}\tG_loss: {:.4f}\n'.format(
        D_total.item() / len(trainloader.dataset),
        G_total.item()*args.critics / len(trainloader.dataset)
    ))
    
    print('D:', class_loss_D)
    print('G:', class_loss_G)
    
    return D_total.item()


if __name__ == "__main__":
    
    # get arguments from command line
    parser = argparse.ArgumentParser(
            description='--> Wassertstein GAN with Gradient Penalty <--')
    parser.add_argument('dataset', type=str,
                        help='the dataset on which to train/test')
    parser.add_argument('--epochs', metavar='N', type=int, default=None,
                        help='number of training epochs (test mode if not specified)')
    parser.add_argument('--batchsize', metavar='N', type=int, default=256,
                        help='number of items in a batch (default 256)')
    parser.add_argument('--dim', metavar='N', type=int, default=64,
                        help='base number of output channels (default 64)')
    parser.add_argument('--critics', metavar='N', type=int, default=5,
                        help='number of discriminator updates per generator update (default 5)')
    parser.add_argument('--z_size', metavar='N', type=int, default=100,
                        help='size of latent space (default 100)')
    parser.add_argument('--load_model', default=False, action='store_true',
                        help='whether to load the model or not (default False)')
    parser.add_argument('--lr', metavar='N', type=float, default=0.0001,
                        help='base learning rate for Adam optimizer (default 0.0001)')
    args = parser.parse_args()
    print(args)
    
    # select device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    # data loaders and class count
    trainloader, validloader, testloader, args.n_classes = get_data_loader(args.dataset, args.batchsize)
    try:
        input_size = trainloader.batch_sampler.sampler.datasource[0][0].shape
    except AttributeError:
        input_size = trainloader.batch_sampler.sampler.data_source[0][0].shape
    
    # prefix for saved model and directory names
    model_prefix = args.dataset + '_'
        
    # classifier to extract features (for fid score computation)
    try:
        classifier = torch.load('classifier.pt', map_location='cpu')
        classifier.eval()
        print('Classifier loaded!')
    except FileNotFoundError:
        classifier = Classifier()
        sys.exit("Need to train a classifier!")
        # TODO: train classifier
        
    # directories for generated samples
    dir_results = model_prefix + 'results'
    dir_samples = model_prefix + 'samples'
    create_directories(dir_results, dir_samples)
    
    
    ########## TEST MODE ##########
    if args.epochs is None:
        # load generator
        G = torch.load(model_prefix + 'generator.pt').to(device)
        print('Generator loaded!')
        # generate samples
        generate_samples(G, dir_samples, args.batchsize, num_samples=4096)
        sampleloader = get_sample_loader(dir_samples, args.batchsize, input_size)
        print('Samples generated!')
        # compute fid score with test set
        fid_score = get_fid_score(classifier, sampleloader, testloader)
        sys.exit("FID score from test set: " + str(fid_score))


    ########## TRAIN MODE ##########
    try:
        if args.load_model:
            G = torch.load(model_prefix + 'generator.pt').to(device)
            D = torch.load(model_prefix + 'discriminator.pt').to(device)
            Model = torch.load(model_prefix + 'model.pt').to(device)
            print('Model loaded!')
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        G = Generator(args.z_size, input_size, args.n_classes, args.dim, args.lr).to(device)
        D = Discriminator(input_size, args.n_classes, args.dim, args.lr).to(device)
        Model = GAN_Model(1234, 1234, args.z_size, args.n_classes).to(device)
        print('Model created!')
        
    # get seeds from Model
    torch.manual_seed(Model.torch_seed)
    np.random.set_state(Model.numpy_seed)
        
    """
    # model summaries
    from torchsummary import summary
    summary(G, (1, args.z_size))
    summary(D, input_size)
    """
    
    ### MAIN TRAINING LOOP ###
    for epoch in np.arange(Model.n_epochs, args.epochs) + 1:
        
        D.train()
        G.train()
        
        # train the model for a single epoch
        epoch_loss = train(G, D, args, trainloader)
        
        # generate a batch of samples
        with torch.no_grad():
            samples = G(Model.sample_batch).cpu()
            samples = samples / 2 + .5
            save_image(samples.view((-1,) + input_size),
                       dir_results + '/samples_' + str(epoch) + '.png',
                       nrow = int(Model.sample_batch.size(0) ** .5))

        # generate samples to compute fid score
        generate_samples(G, dir_samples, args, num_samples=4096)
        sampleloader = get_sample_loader(dir_samples, args.batchsize, input_size)
        
        # compute fid score with validation set
        current_fid = get_fid_score(classifier, sampleloader, validloader)
        
        # if fid score is lower, we save the generator and the discriminator
        if current_fid < Model.best_fid:
            Model.best_fid = current_fid
            torch.save(G, model_prefix + 'generator.pt')
            torch.save(D, model_prefix + 'discriminator.pt')
            print('FID score:', current_fid, '- Model saved!')
        else:
            print('FID score:', current_fid)
        print()
        
        # update and save current state of the model
        Model.n_epochs += 1
        Model.d_losses.append(epoch_loss)
        Model.fid_scores.append(current_fid)
        Model.numpy_seed = np.random.get_state()
        Model.torch_seed = torch.initial_seed()
        torch.save(Model, model_prefix + 'model.pt')
