from torch import nn
from torch.optim import Adam

class Discriminator(nn.Module):
    
    def __init__(self, num_channels, dim=128):
        
        super(Discriminator, self).__init__()
        
        self.num_channels = num_channels
        
        def conv_layer(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))
        
        self.discriminate = nn.Sequential(
            conv_layer(num_channels, dim),
            conv_layer(dim * 1, dim * 2),
            conv_layer(dim * 2, dim * 4),
            conv_layer(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4, stride=1, padding=1, bias=False)
        )
        
        self.optimizer = Adam(self.parameters(), lr=.0001, betas=(.5, .9))
        
    
    def forward(self, x):
        
        out = self.discriminate(x)
        out = out.view([-1, 1])
        
        return out