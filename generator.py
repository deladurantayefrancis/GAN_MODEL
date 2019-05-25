from torch import nn
from torch.optim import Adam

class Generator(nn.Module):
    
    def __init__(self, z_size, input_size, n_classes, dim, lr):
        
        super(Generator, self).__init__()
        n_channels, height, width = input_size
        
        def dconv_layer(in_dim, out_dim, kernel=4, stride=2, padding=1, normalize=True):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding, bias=False),
                nn.Sequential(
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2))
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
        layers.append(dconv_layer((z_size + n_classes), dim, kernel=(height, width), stride=1, padding=0))
        
        # functions called during forward pass
        self.generate = nn.Sequential(*layers[::-1])
        self.tanh = nn.Tanh()
        
        # optimizer
        self.optimizer = Adam(self.parameters(), lr=lr, betas=(.5, .999))
        
    def forward(self, z):
        
        out = z.view([-1, z.size(-1), 1, 1])
        out = self.generate(out)
        
        return self.tanh(out)
    