import torch
import torch.nn as nn
from collections import deque
import vmtk
from monai.networks.blocks import SimpleASPP
from LoGModule import BayesianLoGNN


class BalanceGate(nn.Module):

    def __init__(self, threshold=0.2, queue_size=5, voxel_size=(64, 64, 64)):

        super().__init__()
        self.threshold = threshold
        self.queue_size = queue_size
        self.sq = deque(maxlen=queue_size)
        self.lq = deque(maxlen=queue_size)
        self.voxel_size = voxel_size

    def forward(self, log_output):

        batch_size, channels, depth, height, width = log_output.shape
        boutput = (log_output > 0.5).float()
        pr = boutput.mean(dim=(2, 3, 4))

        for i in range(batch_size):
            if pr[i] < self.threshold:
                self.sq.append(log_output[i])
            else:
                self.lq.append(log_output[i])

        if len(self.sq) == self.queue_size and len(self.lq) == self.queue_size:
            sv = torch.stack(list(self.sq), dim=0)
            lv = torch.stack(list(self.lq), dim=0)
            return torch.cat([sv, lv], dim=0)

        return None


class sLoGNN(nn.Module):

    def __init__(self, in_channels, class_num, base_filter_num=8):
        super().__init__()
        self.base_filter_num = base_filter_num

        self.encoder1 = self._encoder_layer(in_channels, base_filter_num, down_sample=True)
        self.encoder2 = self._encoder_layer(base_filter_num, base_filter_num * 2, down_sample=True)
        self.encoder3 = self._encoder_layer(base_filter_num * 2, base_filter_num * 4, down_sample=True)
        self.encoder4 = self._encoder_layer(base_filter_num * 4, base_filter_num * 8, down_sample=True)

        self.decoder4 = self._decoder_layer(base_filter_num * 8, base_filter_num * 4)
        self.decoder3 = self._decoder_layer(base_filter_num * 4, base_filter_num * 2)
        self.decoder2 = self._decoder_layer(base_filter_num * 2, base_filter_num)
        self.decoder1 = self._decoder_layer(base_filter_num, class_num, final_layer=True)

        self.log_module = BayesianLoGNN()
        self.log_module.initialize_weights()

        self.balance_gate = BalanceGate()

        self.aspp = SimpleASPP(
            spatial_dims=3,
            in_channels=class_num + len(self.log_module.kernel_sizes),
            conv_out_channels=class_num,
        )

    def _encoder_layer(self, in_channels, out_channels, down_sample):

        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2) if down_sample else nn.Identity()
        )

    def _decoder_layer(self, in_channels, out_channels, final_layer=False):

        if final_layer:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

    def forward(self, x):

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4 + e3)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)
        log_features = self.log_module(x)
        gated_output = self.balance_gate(log_features)
        if gated_output is None:
            return None
        combined_features = torch.cat([d1, gated_output], dim=1)
        output = self.aspp(combined_features)
        return output
