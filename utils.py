import numpy as np
import os
import torch
from torch.nn.functional import one_hot
from torchvision.utils import save_image

# selected device (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_directories(dir_results, dir_samples):
    
    try:
        # create a directory for generated image batches
        os.makedirs(dir_results)
        print("Directory", dir_results, "created")
    except FileExistsError:
        print("Directory", dir_results, "already exists")
    
    try:
        # create a directory for generated image samples
        os.makedirs(dir_samples + '/samples')
        print("Directory", dir_samples + '/samples', "created")
    except FileExistsError:
        print("Directory", dir_samples + '/samples', "already exists")
        
        
def get_sample_batch(batchsize, z_size, n_classes, shuffle=True):
    
    samples = torch.randn(batchsize, z_size).to(device)
    classes = np.arange(batchsize, dtype=np.int64) % n_classes
    
    if shuffle:
        np.random.shuffle(classes)
        
    classes = torch.from_numpy(classes).to(device)
    
    return torch.cat([samples, one_hot(classes).float()], dim=1), classes


def generate_samples(G, dir_samples, args, num_samples=1024):
    for i in range(num_samples // args.batchsize):
        with torch.no_grad():
            samples, _ = get_sample_batch(args.batchsize, args.z_size, args.n_classes)
            samples = G(samples).cpu()
            samples = samples / 2 + .5
            for j in range(args.batchsize):
                save_image(samples[j],
                           dir_samples + '/samples/sample_' + str(args.batchsize*i+j) + '.png')
