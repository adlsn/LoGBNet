import numpy as np
import torch
import vmtk
from torch import nn


class BayesianLoGNN(nn.Module):

    def __init__(self, sigma_range=(0.5, 2.5), kernel_sizes=(3, 5, 7, 9, 11)):

        super().__init__()
        self.sigma_range = sigma_range
        self.kernel_sizes = kernel_sizes

        self.log_layers = nn.ModuleList([
            nn.Conv3d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
            for kernel_size in kernel_sizes
        ])
        self.relu = nn.ReLU(inplace=True)

        self.kl_div = 0.0

    def forward(self, x):

        outputs = []
        for layer in self.log_layers:
            outputs.append(self.relu(layer(x)))
        concatenated = torch.cat(outputs, dim=1)

        self.kl_div = self.compute_kl_div()
        return concatenated

    def initialize_weights(self):

        for i, layer in enumerate(self.log_layers):
            kernel = self.load_log_kernel(layer.kernel_size[0])
            layer.weight.data = kernel.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def load_log_kernel(size):

        file_name = f"log_3D_kernel_{size}.npy"
        kernel = np.load(file_name)
        return torch.from_numpy(kernel).float()

    def compute_kl_div(self):

        batch_size, channels, depth, height, width = outputs.size()
        target_distribution = torch.full_like(outputs, 1.0 / (depth * height * width))

        predicted_distribution = F.softmax(outputs, dim=1)

        epsilon = 1e-10
        predicted_distribution = torch.clamp(predicted_distribution, epsilon, 1.0)
        target_distribution = torch.clamp(target_distribution, epsilon, 1.0)

        kl_div = torch.sum(
            target_distribution * torch.log(target_distribution / predicted_distribution)
        ) / batch_size

        return kl_div.item()
