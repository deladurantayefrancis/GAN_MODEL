#!/usr/bin/python
import argparse
import numpy as np
import os
import sys

import torch
from torch.nn.functional import one_hot
from torchvision.utils import save_image

from classifier import Classifier
from discriminator import Discriminator
from generator import Generator

from loader import get_data_loader
from score_fid import get_sample_loader
from score_fid import get_fid_score


class GAN_Model(torch.nn.Module):
    def __init__(self, numpy_initial_seed, torch_initial_seed, z_size, n_classes):
        super(GAN_Model, self).__init__()
        # set and save initial seeds for reproducibility
        np.random.seed(numpy_initial_seed)
        torch.manual_seed(torch_initial_seed)
        self.numpy_seed = np.random.get_state()
        self.torch_seed = torch.initial_seed()
        # save best fid and training epochs count
        self.best_fid = float('Inf')
        self.n_epochs = 0
        self.d_losses = []
        # batch of samples for visuals creation
        self.sample_batch, _ = get_sample_batch(64, z_size, n_classes)


def create_directories(dir_results, dir_samples):
    
    try:
        # Create target Directory
        os.makedirs(dir_results)
        print("Directory", dir_results, "Created")
    except FileExistsError:
        print("Directory", dir_results, "already exists")
    
    try:
        # Create target Directory
        os.makedirs(dir_samples + '/samples')
        print("Directory", dir_samples + '/samples', "Created")
    except FileExistsError:
        print("Directory", dir_samples + '/samples', "already exists")
        
        
def get_sample_batch(batchsize, z_size, n_classes):
    
    samples = torch.randn(batchsize, z_size).to(device)
    classes = one_hot(torch.randint(0, n_classes, [batchsize])).to(device)
    
    return torch.cat([samples, classes.float()], dim=1), classes


def train(G, D, args, trainloader):
    
    D_total = 0
    G_total = 0
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        
        D.optimizer.zero_grad()
        
        # get real images with classes
        real_images = data.to(device)
        real_labels = one_hot(labels).to(device)
        
        # get fake images with classes
        samples, fake_labels = get_sample_batch(data.size(0), args.z_size, args.n_classes)
        fake_images = G(samples)
        
        # compute gradient penalty
        alpha = torch.rand([data.shape[0], 1, 1, 1]).to(device)
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images
        interpolated_scores, _ = D(interpolated_images)
        interpolated_grad = torch.autograd.grad(
            outputs = interpolated_scores,
            inputs = interpolated_images,
            grad_outputs = torch.ones([data.shape[0], 1]).to(device),
            create_graph = True,
            retain_graph = True,
            only_inputs = True)[0]
        grad_norm = torch.norm(interpolated_grad, 2, dim=1)
        grad_penalty = 10 * torch.mean((grad_norm - 1) ** 2)
        
        # wasserstein loss
        real_scores, real_logits = D(real_images)
        fake_scores, fake_logits = D(fake_images)
        real_loss = torch.mean(real_scores)
        fake_loss = torch.mean(fake_scores)
        wgan_loss = fake_loss - real_loss
        
        # class prediction loss
        criterion = torch.nn.BCELoss()
        real_class_loss = criterion(real_logits, real_labels.float())
        fake_class_loss = criterion(fake_logits, fake_labels.float())
        class_loss = real_class_loss + fake_class_loss
        
        # discriminator loss
        D_loss = wgan_loss + class_loss + grad_penalty
        D_loss.backward()
        D.optimizer.step()
        
        D_total += D_loss * data.shape[0]
        
        # compute Generator loss
        if batch_idx % args.critics == 0:
            
            G.optimizer.zero_grad()
            
            samples, _ = get_sample_batch(data.size(0), args.z_size, args.n_classes)
            fake_scores, _ = D(G(samples))
            
            G_loss = -torch.mean(fake_scores)
            G_loss.backward()
            G.optimizer.step()
            
            G_total += G_loss * data.shape[0]
        
        # batch info display
        if (batch_idx + 1) % int(len(trainloader.dataset) / len(data) / 9) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD_loss: {:.6f}\tG_loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                D_loss.item(), G_loss.item()))
    
    print('wgan_loss =', wgan_loss.item())
    print('====> Average epoch loss:\tD_loss: {:.4f}\tG_loss: {:.4f}\n'.format(
        D_total.item() / len(trainloader.dataset),
        G_total.item()*args.critics / len(trainloader.dataset)
    ))
    
    return D_total.item()
    
    
def generate_samples(G, dir_samples, args, num_samples=1024):
    for i in range(num_samples // args.batchsize):
        with torch.no_grad():
            samples, _ = get_sample_batch(args.batchsize, args.z_size, args.n_classes)
            samples = G(samples).cpu()
            samples = samples / 2 + .5
            for j in range(args.batchsize):
                save_image(samples[j],
                           dir_samples + '/samples/sample_' + str(args.batchsize*i+j) + '.png')


if __name__ == "__main__":
    
    # get arguments from command line
    parser = argparse.ArgumentParser(
            description='--> Wassertstein GAN with Gradient Penalty <--')
    parser.add_argument('dataset', type=str,
                        help='the dataset on which to train/test')
    parser.add_argument('--epochs', metavar='N', type=int, default=None,
                        help='number of training epochs (test mode if not specified)')
    parser.add_argument('--batchsize', metavar='N', type=int, default=32,
                        help='number of items in a batch (default 32)')
    parser.add_argument('--dim', metavar='N', type=int, default=64,
                        help='base number of output channels (default 64)')
    parser.add_argument('--critics', metavar='N', type=int, default=3,
                        help='number of discriminator update per generator update (default 5)')
    parser.add_argument('--z_size', metavar='N', type=int, default=100,
                        help='size of latent space (default 100)')
    parser.add_argument('--load_model', default=False, action='store_true',
                        help='whether to load the model or not (default False)')
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
        G = Generator(input_size, args.n_classes, args.z_size, args.dim).to(device)
        D = Discriminator(input_size, args.n_classes, args.dim).to(device)
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
                       dir_results + '/samples_' + str(epoch) + '.png')

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
        Model.numpy_seed = np.random.get_state()
        Model.torch_seed = torch.initial_seed()
        torch.save(Model, model_prefix + 'model.pt')
