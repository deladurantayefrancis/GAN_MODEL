from torch import nn
from torch.optim import Adam

class Generator(nn.Module):
    
    def __init__(self, z_size, input_size, dim=64):
        
        super(Generator, self).__init__()
        
        self.best_fid = float('Inf')
        self.n_epochs = 0
        
        def dconv_layer(in_dim, out_dim, kernel=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        
        # last layer
        layers = [nn.ConvTranspose2d(dim, input_size[0], 4, stride=2, padding=1, bias=False)]
        height = input_size[1] // 2
        width = input_size[2] // 2
        
        # middle layers
        while height > 4 or width > 4:
            layers.insert(0, dconv_layer(dim*2, dim))
            height //= 2
            width //= 2
            dim *= 2
            
        # first layer
        layers.insert(0, dconv_layer(z_size, dim, kernel=(height, width), stride=1, padding=0))
        
        # functions called during forward pass
        self.generate = nn.Sequential(*layers)
        self.tanh = nn.Tanh()
        
        # optimizer
        self.optimizer = Adam(self.parameters(), lr=.00005, betas=(.5, .9))
        
        
    def forward(self, z):
        
        out = z.view([-1, z.size(-1), 1, 1])
        out = self.generate(out)
        
        return self.tanh(out)
    