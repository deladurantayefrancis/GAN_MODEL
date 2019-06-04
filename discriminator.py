from torch import nn
from torch.optim import Adam

class Discriminator(nn.Module):
    
    def __init__(self, input_size, n_classes, dim, lr):
        
        super(Discriminator, self).__init__()
        n_channels, height, width = input_size
        
        def conv_layer(in_dim, out_dim, downsampling=False):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1, bias=False) if downsampling
                    else nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5))
        
        # first layer
        layers = [conv_layer(n_channels, dim, downsampling=True)]
        height //= 2
        width //= 2
        
        # middle layers
        while height > 4 or width > 4:
            layers.append(conv_layer(dim, dim))
            layers.append(conv_layer(dim, dim*2, downsampling=True))
            height //= 2
            width //= 2
            dim *= 2
        
        # last layer
        layers.append(nn.Conv2d(dim, n_classes + 1, (height, width), bias=False))
        
        # functions called during forward pass
        self.discriminate = nn.Sequential(*layers)
        
        # optimizer
        self.optimizer = Adam(self.parameters(), lr=lr, betas=(.5, .999))
        
    
    def forward(self, x):
        
        out = self.discriminate(x)
        out = out.squeeze()
        
        scores = out[:, 0]
        logits = out[:, 1:]
        
        return scores.view(-1, 1), logits
    