from torch import nn
from torch.optim import Adam

class Generator(nn.Module):
    
    def __init__(self, z_size, input_size, n_classes, dim, lr):
        
        super(Generator, self).__init__()
        n_channels, height, width = input_size
        
        def dconv_layer(in_dim, out_dim, normalize=True):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim, out_dim, 3, 1, 0, bias=False),
                nn.Sequential(
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU())
                if normalize else nn.Identity())
            
        # last layer
        layers = [dconv_layer(dim, n_channels, normalize=False)]
        height //= 2
        width //= 2
        
        # middle layers
        while height > 4 or width > 4:
            layers.append(dconv_layer(dim*2, dim))
            height //= 2
            width //= 2
            dim *= 2
        
        # first layer
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(z_size + n_classes, dim, (height, width), bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()))
        
        # functions called during forward pass
        self.generate = nn.Sequential(*layers[::-1])
        self.tanh = nn.Tanh()
        
        # optimizer
        self.optimizer = Adam(self.parameters(), lr=lr, betas=(.5, .999))
        
    def forward(self, z):
        
        out = z.view([-1, z.size(-1), 1, 1])
        out = self.generate(out)
        
        return self.tanh(out)
    