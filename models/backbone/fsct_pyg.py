"""FSCT (Fast Semantic and Instance Segmentation) implementation using PyTorch Geometric.

Reference:
Fast Semantic and Instance Segmentation of 3D Point Clouds
https://github.com/huixiancheng/FSCT
"""

from typing import List, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN

from torch_geometric.nn import (
    PointNetConv,
    fps,
    radius,
    global_max_pool,
    knn_interpolate,
)

from ..build import MODELS


def MLP(channels, batch_norm=True):
    """Create a multi-layer perceptron with batch normalization and ReLU activation."""
    if batch_norm:
        return Seq(
            *[
                Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
                for i in range(1, len(channels))
            ]
        )
    else:
        return Seq(
            *[
                Seq(Lin(channels[i - 1], channels[i]), ReLU())
                for i in range(1, len(channels))
            ]
        )


class SAModule(torch.nn.Module):
    """Set Abstraction Module with PointNetConv."""

    def __init__(self, ratio, r, NN):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(NN)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """Global Set Abstraction Module with max pooling."""

    def __init__(self, NN):
        super(GlobalSAModule, self).__init__()
        self.NN = NN

    def forward(self, x, pos, batch):
        x = self.NN(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    """Feature Propagation Module with k-NN interpolation."""

    def __init__(self, k, NN):
        super(FPModule, self).__init__()
        self.k = k
        self.NN = NN

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.NN(x)
        return x, pos_skip, batch_skip


@MODELS.register_module()
class FSCTEncoderPyG(nn.Module):
    """FSCT Encoder using PyG for variable-size point clouds.

    Args:
        in_channels: Input feature dimension (default: 3 for XYZ)
        num_classes: Number of output classes (not used in encoder)
        sa1_ratio: Sampling ratio for SA1 module (default: 0.1)
        sa1_radius: Radius for SA1 module (default: 0.2)
        sa1_mlp: MLP channels for SA1 (default: [3, 128, 256, 512])
        sa2_ratio: Sampling ratio for SA2 module (default: 0.05)
        sa2_radius: Radius for SA2 module (default: 0.4)
        sa2_mlp: MLP channels for SA2 (default: [512+3, 512, 1024, 1024])
        sa3_mlp: MLP channels for SA3 global module (default: [1024+3, 1024, 2048, 2048])
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = None,
        sa1_ratio: float = 0.1,
        sa1_radius: float = 0.2,
        sa1_mlp: List[int] = None,
        sa2_ratio: float = 0.05,
        sa2_radius: float = 0.4,
        sa2_mlp: List[int] = None,
        sa3_mlp: List[int] = None,
        **kwargs,
    ):
        super(FSCTEncoderPyG, self).__init__()

        # Default MLP configurations
        # SA1: PointNetConv concatenates [x_j, pos_j - pos_i] = [in_channels, 3]
        if sa1_mlp is None:
            sa1_mlp = [in_channels + 3, 128, 256, 512]
        else:
            sa1_mlp = [in_channels + 3] + sa1_mlp

        if sa2_mlp is None:
            sa2_mlp = [512 + 3, 512, 1024, 1024]
        else:
            sa2_mlp = [sa1_mlp[-1] + in_channels] + sa2_mlp

        if sa3_mlp is None:
            sa3_mlp = [1024 + 3, 1024, 2048, 2048]
        else:
            sa3_mlp = [sa2_mlp[-1] + in_channels] + sa3_mlp

        self.in_channels = in_channels

        # Set Abstraction modules
        self.sa1_module = SAModule(sa1_ratio, sa1_radius, MLP(sa1_mlp))
        self.sa2_module = SAModule(sa2_ratio, sa2_radius, MLP(sa2_mlp))
        self.sa3_module = GlobalSAModule(MLP(sa3_mlp))

        # Output channel information
        self.out_channels = sa3_mlp[-1]
        self.channel_list = [in_channels, sa1_mlp[-1], sa2_mlp[-1], sa3_mlp[-1]]

        logging.info(
            f"FSCTEncoderPyG initialized with channel_list: {self.channel_list}"
        )

    def forward_seg_feat(self, p, x=None, b=None):
        """Forward pass returning feature pyramids for segmentation.

        Args:
            p: Point positions (N, 3) or dict with 'pos', 'x', 'batch'
            x: Point features (N, C) or None
            b: Batch indices (N,) or None

        Returns:
            l_p: List of positions at each scale
            l_x: List of features at each scale
            l_b: List of batch indices at each scale
        """
        if hasattr(p, "keys"):
            p, x, b = p["pos"], p.get("x"), p["batch"]

        if x is None:
            x = p.clone()
        if b is None:
            b = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)

        # Initial scale (SA0)
        sa0_out = (x, p, b)

        # SA1
        sa1_out = self.sa1_module(*sa0_out)

        # SA2
        sa2_out = self.sa2_module(*sa1_out)

        # SA3 (global)
        sa3_out = self.sa3_module(*sa2_out)

        # Collect outputs
        l_x = [sa0_out[0], sa1_out[0], sa2_out[0], sa3_out[0]]
        l_p = [sa0_out[1], sa1_out[1], sa2_out[1], sa3_out[1]]
        l_b = [sa0_out[2], sa1_out[2], sa2_out[2], sa3_out[2]]

        return l_p, l_x, l_b

    def forward(self, p, x=None, b=None):
        """Forward pass."""
        return self.forward_seg_feat(p, x, b)


@MODELS.register_module()
class FSCTDecoderPyG(nn.Module):
    """FSCT Decoder using PyG for variable-size point clouds.

    Args:
        encoder_channel_list: List of channel dimensions from encoder
        fp3_mlp: MLP channels for FP3 (default: [3072, 1024, 1024])
        fp2_mlp: MLP channels for FP2 (default: [1536, 1024, 1024])
        fp1_mlp: MLP channels for FP1 (default: [1024, 1024, 1024])
        fp3_k: k for FP3 k-NN interpolation (default: 1)
        fp2_k: k for FP2 k-NN interpolation (default: 3)
        fp1_k: k for FP1 k-NN interpolation (default: 3)
    """

    def __init__(
        self,
        encoder_channel_list: List[int],
        fp3_mlp: List[int] = None,
        fp2_mlp: List[int] = None,
        fp1_mlp: List[int] = None,
        fp3_k: int = 1,
        fp2_k: int = 3,
        fp1_k: int = 3,
        **kwargs,
    ):
        super(FSCTDecoderPyG, self).__init__()

        # Default configurations based on FSCT architecture
        # FP3: [2048 + 1024 = 3072, 1024, 1024]
        # FP2: [1024 + 512 = 1536, 1024, 1024]
        # FP1: [1024 + input_channels = 1024, 1024, 1024]

        if fp3_mlp is None:
            fp3_mlp = [encoder_channel_list[3] + encoder_channel_list[2], 1024, 1024]

        if fp2_mlp is None:
            fp2_mlp = [1024 + encoder_channel_list[1], 1024, 1024]

        if fp1_mlp is None:
            fp1_mlp = [1024 + encoder_channel_list[0], 1024, 1024]

        # Feature Propagation modules
        self.fp3_module = FPModule(fp3_k, MLP(fp3_mlp))
        self.fp2_module = FPModule(fp2_k, MLP(fp2_mlp))
        self.fp1_module = FPModule(fp1_k, MLP(fp1_mlp))

        self.out_channels = fp1_mlp[-1]

        logging.info(
            f"FSCTDecoderPyG initialized with out_channels: {self.out_channels}"
        )
        logging.info(f"FP MLPs: fp3={fp3_mlp}, fp2={fp2_mlp}, fp1={fp1_mlp}")

    def forward(self, l_p, l_x, l_b):
        """Forward pass for decoder.

        Args:
            l_p: List of positions [(N_0, 3), ..., (N_k, 3)]
            l_x: List of features [(N_0, C_0), ..., (N_k, C_k)]
            l_b: List of batch indices [(N_0,), ..., (N_k,)]

        Returns:
            x: (N_0, C_out) final features at original resolution
        """
        # Unpack scales (SA0, SA1, SA2, SA3)
        sa0_x, sa0_p, sa0_b = l_x[0], l_p[0], l_b[0]
        sa1_x, sa1_p, sa1_b = l_x[1], l_p[1], l_b[1]
        sa2_x, sa2_p, sa2_b = l_x[2], l_p[2], l_b[2]
        sa3_x, sa3_p, sa3_b = l_x[3], l_p[3], l_b[3]

        # FP3: SA3 -> SA2
        fp3_x, _, _ = self.fp3_module(sa3_x, sa3_p, sa3_b, sa2_x, sa2_p, sa2_b)

        # FP2: FP3 -> SA1
        fp2_x, _, _ = self.fp2_module(fp3_x, sa2_p, sa2_b, sa1_x, sa1_p, sa1_b)

        # FP1: FP2 -> SA0
        fp1_x, _, _ = self.fp1_module(fp2_x, sa1_p, sa1_b, sa0_x, sa0_p, sa0_b)

        return fp1_x


@MODELS.register_module()
class FSCTNet(nn.Module):
    """Complete FSCT Network for semantic segmentation.

    This is a standalone version that includes the final classification head.
    For use with VariableSeg model, use FSCTEncoderPyG and FSCTDecoderPyG instead.

    Args:
        num_classes: Number of output classes
        in_channels: Input feature dimension (default: 3 for XYZ)
        dropout: Dropout rate for final classifier (default: 0.5)
    """

    def __init__(
        self, num_classes: int, in_channels: int = 3, dropout: float = 0.5, **kwargs
    ):
        super(FSCTNet, self).__init__()

        # Encoder
        self.encoder = FSCTEncoderPyG(in_channels=in_channels, **kwargs)

        # Decoder
        self.decoder = FSCTDecoderPyG(
            encoder_channel_list=self.encoder.channel_list, **kwargs
        )

        # Final classification layers
        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, num_classes, 1)
        self.drop1 = torch.nn.Dropout(dropout)
        self.bn1 = torch.nn.BatchNorm1d(1024)

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyG Data object with pos, x (optional), batch

        Returns:
            x: (batch_size, num_classes, num_points) logits
        """
        # Encoder
        l_p, l_x, l_b = self.encoder(data.pos, data.x, data.batch)

        # Decoder
        x = self.decoder(l_p, l_x, l_b)

        # Reshape for 1D convolution (B, C, N)
        x = x.unsqueeze(dim=0)  # (1, N, C)
        x = x.permute(0, 2, 1)  # (1, C, N)

        # Final classification
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)

        return x
