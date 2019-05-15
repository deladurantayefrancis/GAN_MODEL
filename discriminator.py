from torch import nn
from torch.optim import Adam

class Discriminator(nn.Module):
    
    def __init__(self, input_size, dim=64):
        
        super(Discriminator, self).__init__()
        
        def conv_layer(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))
        
        # first layer
        layers = [conv_layer(input_size[0], dim)]
        height = input_size[1] // 2
        width = input_size[2] // 2
        
        # middle layers
        while height > 4 or width > 4:
            layers.append(conv_layer(dim, dim*2))
            height //= 2
            width //= 2
            dim *= 2
        
        # last layer
        layers.append(nn.Conv2d(dim, 1, (height, width), bias=False))
        
        # function called during forward pass
        self.discriminate = nn.Sequential(*layers)
        
        # optimizer
        self.optimizer = Adam(self.parameters(), lr=.00005, betas=(.5, .9))
        
    
    def forward(self, x):
        
        out = self.discriminate(x)
        out = out.view([-1, 1])
        
        return out
    