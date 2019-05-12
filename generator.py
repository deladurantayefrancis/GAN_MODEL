from torch import nn
from torch.optim import Adam

class Generator(nn.Module):
    
    def __init__(self, z_size, num_channels, dim=128):
        
        super(Generator, self).__init__()
        
        self.z_size = z_size
        
        self.best_fid = float('Inf')
        self.n_epochs = 0
        
        def dconv_layer(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        
        self.generate = nn.Sequential(
            #nn.ConvTranspose2d(z_size, dim * 8, 4, stride=2, padding=1, bias=False),
            dconv_layer(z_size, dim * 8),
            dconv_layer(dim * 8, dim * 4),
            dconv_layer(dim * 4, dim * 2),
            dconv_layer(dim * 2, dim * 1),
            nn.ConvTranspose2d(dim, num_channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        #self.tanh = nn.Tanh()
        
        self.optimizer = Adam(self.parameters(), lr=.0001, betas=(.5, .9))
        
        
    def forward(self, z):
        
        out = z.view([-1, self.z_size, 1, 1])
        out = self.generate(out)
        
        return out