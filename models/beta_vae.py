import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class BetaVAE(nn.Module):
    def __init__(self, num_classes = None, pretrained=False, latent_dim=20, beta=4.0):
        super().__init__()
        self.beta = beta
        self.z_dim = 20 

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.ReLU(True),
            View((-1,256*2*2)),                 # B, 256
            nn.Linear(256*2*2, self.z_dim*2),             # B, z_dim*2

        )
     
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256*2*2),               # B, 256
            View((-1, 256, 2, 2)),   
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 2 → 4
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 4 → 8
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 8 → 16
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 16 → 32
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    def encode(self, x):
        return self.encoder(x)
    
    def encode_mu(self, x):
        x = self.encoder(x)
        mu = x[:, :self.z_dim]
        return mu


    def decode(self, z):
        return self.decoder(z)

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = reconstruction_loss(x, x_recon, "gaussian")
        kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        return recon_loss + self.beta * kld, recon_loss, self.beta * kld