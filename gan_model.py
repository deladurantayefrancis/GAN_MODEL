#!/usr/bin/python
import argparse
import numpy as np
import os
import sys

import torch
from torchvision.utils import save_image

from classifier import Classifier
from discriminator import Discriminator
from generator import Generator

from loader import get_data_loader
from score_fid import get_sample_loader
from score_fid import get_fid_score


class GAN_Model(torch.nn.Module):
    def __init__(self, numpy_initial_seed, torch_initial_seed):
        super(GAN_Model, self).__init__()
        # set and save initial seeds for reproducibility
        np.random.seed(numpy_initial_seed)
        torch.manual_seed(torch_initial_seed)
        self.numpy_seed = np.random.get_state()
        self.torch_seed = torch.initial_seed()
        # save best fid and training epochs count
        self.best_fid = float('Inf')
        self.n_epochs = 0
        # batch of samples for visuals creation
        self.sample_batch = torch.randn(64, z_size).to(device)


def create_directories(dirResults, dirSamples):
    
    try:
        # Create target Directory
        os.makedirs(dirResults)
        print("Directory", dirResults, "Created")
    except FileExistsError:
        print("Directory", dirResults, "already exists")
    
    try:
        # Create target Directory
        os.makedirs(dirSamples + '/samples')
        print("Directory", dirSamples + '/samples', "Created")
    except FileExistsError:
        print("Directory", dirSamples + '/samples', "already exists")


def train(G, D, n_critics, trainloader):
    
    D.train()
    G.train()
    
    D_total = 0
    G_total = 0
    
    for batch_idx, (data, _) in enumerate(trainloader):
        
        x_real = data.to(device)
        
        D.optimizer.zero_grad()
        G.optimizer.zero_grad()
        
        # generate fake image
        sample = torch.randn(data.shape[0], z_size).to(device)
        x_fake = G(sample)
        
        # compute gradient penalty
        alpha = torch.rand([data.shape[0], 1, 1, 1]).to(device)
        interpolated = alpha * x_real + (1 - alpha) * x_fake
        d_interpolated = D(interpolated)
        interpolated_grad = torch.autograd.grad(
            outputs = d_interpolated,
            inputs = interpolated,
            grad_outputs = torch.ones([data.shape[0], 1]).to(device),
            create_graph = True,
            retain_graph = True,
            only_inputs = True)[0]
        grad_norm = torch.norm(interpolated_grad, 2, dim=1)
        grad_penalty = 10 * torch.mean((grad_norm - 1) ** 2)
        
        # compute Wasserstein and Discriminator loss
        d_real = D(x_real)
        d_fake = D(x_fake)
        loss_real = torch.mean(d_real)
        loss_fake = torch.mean(d_fake)
        
        wasserstein_loss = loss_fake - loss_real
        D_loss = wasserstein_loss + grad_penalty
        D_loss.backward()
        D.optimizer.step()
        
        D_total += D_loss * data.shape[0]
        
        # compute Generator loss
        if batch_idx % n_critics == 0:
            
            D.optimizer.zero_grad()
            G.optimizer.zero_grad()
            
            sample = torch.randn(data.shape[0], z_size).to(device)
            d_fake = D(G(sample))
            
            G_loss = -torch.mean(d_fake)
            G_loss.backward()
            G.optimizer.step()
            
            G_total += G_loss * data.shape[0]
        
        # batch info display
        if (batch_idx + 1) % int(len(trainloader.dataset) / len(data) / 9) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD_loss: {:.6f}\tG_loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                D_loss.item(), G_loss.item()))
    
    print('wasserstein_loss =', wasserstein_loss.item())
    print('====> Average epoch loss:\tD_loss: {:.4f}\tG_loss: {:.4f}\n'.format(
        D_total.item() / len(trainloader.dataset),
        G_total.item()*n_critics / len(trainloader.dataset)
    ))
    
    
def generate_samples(G, dirSamples, batch_size, num_samples=1024):
    for i in range(num_samples // args.batch_size):
        with torch.no_grad():
            samples = torch.randn(batch_size, z_size).to(device)
            samples = G(samples).cpu()
            samples = samples / 2 + .5
            for j in range(batch_size):
                save_image(samples[j],
                           dirSamples + '/samples/sample_' + str(batch_size*i+j) + '.png')


if __name__ == "__main__":
    
    # get arguments from command line
    parser = argparse.ArgumentParser(
            description='--> Wassertstein GAN with Gradient Penalty <--')
    parser.add_argument('dataset', type=str,
                        help='the dataset on which to train/test')
    parser.add_argument('--epochs', metavar='N', type=int, default=None,
                        help='number of training epochs (test mode if not specified)')
    parser.add_argument('--batch_size', metavar='N', type=int, default=32,
                        help='number of items in a batch (default 32)')
    parser.add_argument('--dim', metavar='N', type=int, default=64,
                        help='base number of output channels (default 64)')
    args = parser.parse_args()
    print(args)
    
    # select device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # hyperparameters
    n_critics = 5
    z_size = 100
    
    # set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    # data loaders
    trainloader, validloader, testloader = get_data_loader(args.dataset, args.batch_size)
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
    dirResults = model_prefix + 'results'
    dirSamples = model_prefix + 'samples'
    create_directories(dirResults, dirSamples)
    
    
    ########## TEST MODE ##########
    if args.epochs is None:
        # load generator
        G = torch.load(model_prefix + 'generator.pt').to(device)
        print('Generator loaded!')
        # generate samples
        generate_samples(G, dirSamples, args.batch_size, num_samples=4096)
        sampleloader = get_sample_loader(dirSamples, args.batch_size, input_size)
        print('Samples generated!')
        # compute fid score with test set
        fid_score = get_fid_score(classifier, sampleloader, testloader)
        sys.exit("FID score from test set: " + str(fid_score))


    ########## TRAIN MODE ##########
    try:
        G = torch.load(model_prefix + 'generator.pt').to(device)
        D = torch.load(model_prefix + 'discriminator.pt').to(device)
        Model = torch.load(model_prefix + 'model.pt').to(device)
        print('Model loaded!')
    except FileNotFoundError:
        G = Generator(z_size, input_size, args.dim).to(device)
        D = Discriminator(input_size, args.dim).to(device)
        Model = GAN_Model(1234, 1234).to(device)
        print('Model created!')
        
    # get seeds from Model
    torch.manual_seed(Model.torch_seed)
    np.random.set_state(Model.numpy_seed)
        
    """
    # model summaries
    from torchsummary import summary
    summary(G, (1, z_size))
    summary(D, input_size)
    """
    
    ### MAIN TRAINING LOOP ###
    for epoch in np.arange(Model.n_epochs, args.epochs) + 1:
        
        # train the model for a single epoch
        train(G, D, n_critics, trainloader)
        
        # generate a batch of samples
        with torch.no_grad():
            samples = G(Model.sample_batch).cpu()
            samples = samples / 2 + .5
            save_image(samples.view((-1,) + input_size),
                       dirResults + '/samples_' + str(epoch) + '.png')

        # generate samples to compute fid score
        generate_samples(G, dirSamples, args.batch_size, num_samples=4096)
        sampleloader = get_sample_loader(dirSamples, args.batch_size, input_size)
        
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
        Model.numpy_seed = np.random.get_state()
        Model.torch_seed = torch.initial_seed()
        torch.save(Model, model_prefix + 'model.pt')
