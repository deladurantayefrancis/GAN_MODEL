from torch import nn
from torch.optim import Adam

class Discriminator(nn.Module):
    
    def __init__(self, input_size, n_classes, dim, lr):
        
        super(Discriminator, self).__init__()
        n_channels, height, width = input_size
        
        def conv_layer(in_dim, out_dim, kernel=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel, stride, padding, bias=False),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2))
        
        # first layer
        layers = [conv_layer(n_channels, dim)]
        height //= 2
        width //= 2
        
        # middle layers
        while height > 4 or width > 4:
            layers.append(conv_layer(dim, dim*2))
            height //= 2
            width //= 2
            dim *= 2
        
        # last layer
        layers.append(nn.Conv2d(dim, n_classes + 1, (height, width), bias=False))
        
        # functiona called during forward pass
        self.discriminate = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
        
        # optimizer
        self.optimizer = Adam(self.parameters(), lr=lr, betas=(.5, .999))
        
    
    def forward(self, x):
        
        out = self.discriminate(x)
        out = out.squeeze()
        
        scores = out[:, 0]
        logits = self.softmax(out[:, 1:])
        
        return scores.view(-1, 1), logits
    