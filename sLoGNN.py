import torch
import torch.nn as nn
import vmtk
from monai.networks.blocks import SimpleASPP
from LoGModule import BayesianLoGNN


class sLoGNN(nn.Module):

    def __init__(self, in_channels, class_num, base_filter_num=8):
        super().__init__()
        self.base_filter_num = base_filter_num

        # Encoder layers
        self.encoder1 = self._encoder_layer(in_channels, base_filter_num, down_sample=True)
        self.encoder2 = self._encoder_layer(base_filter_num, base_filter_num * 2, down_sample=True)
        self.encoder3 = self._encoder_layer(base_filter_num * 2, base_filter_num * 4, down_sample=True)
        self.encoder4 = self._encoder_layer(base_filter_num * 4, base_filter_num * 8, down_sample=True)

        # Decoder layers
        self.decoder4 = self._decoder_layer(base_filter_num * 8, base_filter_num * 4)
        self.decoder3 = self._decoder_layer(base_filter_num * 4, base_filter_num * 2)
        self.decoder2 = self._decoder_layer(base_filter_num * 2, base_filter_num)
        self.decoder1 = self._decoder_layer(base_filter_num, class_num, final_layer=True)

        # LoG stream
        self.log_module = BayesianLoGNN()
        self.log_module.initialize_weights()

        # ASPP layer for feature refinement
        self.aspp = SimpleASPP(
            spatial_dims=3,
            in_channels=class_num + len(self.log_module.kernel_sizes),
            conv_out_channels=class_num,
        )

    def _encoder_layer(self, in_channels, out_channels, down_sample):
        """
        Create an encoder layer block.
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2) if down_sample else nn.Identity()
        )

    def _decoder_layer(self, in_channels, out_channels, final_layer=False):
        """
        Create a decoder layer block.
        """
        if final_layer:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Encode
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decode
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4 + e3)
        d2 = self.decoder2(d3 + e2)
        d1 = self.decoder1(d2 + e1)

        # LoG processing
        log_features = self.log_module(x)

        # Combine LoG and decoder outputs
        combined_features = torch.cat([d1, log_features], dim=1)
        output = self.aspp(combined_features)
        return output
