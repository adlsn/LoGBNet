import torch
import torch.nn as nn
import vmtk
import torch.nn.functional as F
from torch.distributions import Normal


class BayesianConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, prior_std=1.0, kf=None):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.prior_std = prior_std

        self.weight_mu = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.full_like(self.weight_mu, -3.0)) 

        if kf:
            self.initializeLK(kf)

        self.prior = Normal(0, prior_std)

    def forward(self, x):

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))  
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)  

        return F.conv3d(x, weight, padding=self.kernel_size // 2)

    def kl_divergence(self):

        posterior = Normal(self.weight_mu, torch.log1p(torch.exp(self.weight_rho)))

        kl = torch.distributions.kl_divergence(posterior, self.prior)
        return kl.sum()

    def initializeLK(self, kf):

        lk = torch.from_numpy(np.load(kf)).float() 
        if lk.shape == self.weight_mu.shape:
            with torch.no_grad():
                self.weight_mu.copy_(lk)
        else:
            raise ValueError(f"Error!!!")

class BayesianLoGNN(nn.Module):
    def __init__(self, sigma_range=(0.5, 2.5), kernel_sizes=(3, 5, 7, 9, 11), kernel_dir="log_kernels/"):

        super().__init__()
        self.sigma_range = sigma_range
        self.kernel_sizes = kernel_sizes

        self.log_layers = nn.ModuleList([
            BayesianConv3D(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                kernel_file=f"{kernel_dir}/log_3D_kernel_{kernel_size}.npy"
            )
            for kernel_size in kernel_sizes
        ])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []
        for layer in self.log_layers:
            outputs.append(self.relu(layer(x)))
        concatenated = torch.cat(outputs, dim=1)
        return concatenated

    def compute_kl_div(self):

        total_kl = sum(layer.kl_divergence() for layer in self.log_layers)
        return total_kl
