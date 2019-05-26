import numpy as np
import torch

from utils import get_sample_batch

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
        
        # save discriminator losses and validation fid scores during traning
        self.d_losses = []
        self.fid_scores = []
        
        # batch of samples for visuals creation
        self.sample_batch, _ = get_sample_batch(100, z_size, n_classes, shuffle=False)