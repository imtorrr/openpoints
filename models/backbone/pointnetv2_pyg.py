"""PointNet++ implementation using PyTorch Geometric for variable-size inputs.

Reference:
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
by Charles R. Qi, Li (Eric) Yi, Hao Su, Leonidas J. Guibas
https://github.com/sshaoshuai/Pointnet2.PyTorch
"""

from typing import List, Optional
import logging
import torch
import torch.nn as nn

from torch_geometric.nn import (
    radius,
    knn,
    fps,
    knn_interpolate,
)
from torch_scatter import scatter

from ..build import MODELS
from ..layers import create_linearblock, create_act, CHANNEL_MAP, random_sample


def create_grouper(group_args, support_same_as_query=True):
    """Create a grouping function for neighborhood queries."""
    method = group_args.get("NAME", "ballquery").lower()
    radius_val = group_args.get("radius", 0.1)
    nsample = group_args.get("nsample", 16)

    logging.info(f"Grouper: {group_args}")
    if method in ["ballquery", "ball", "query"]:
        if support_same_as_query:
            grouper = lambda p, b: radius(
                p,
                p,
                radius_val,
                b,
                b,
                max_num_neighbors=nsample,
                flow="target_to_source",
            )
        else:
            grouper = lambda p_support, p_query, b_support, b_query: radius(
                p_support,
                p_query,
                radius_val,
                b_support,
                b_query,
                max_num_neighbors=nsample,
            )
    elif method in ["knn", "knn_graph"]:
        if support_same_as_query:
            grouper = lambda p, b: knn(p, p, nsample, b, b, flow="target_to_source")
        else:
            grouper = lambda p_support, p_query, b_support, b_query: knn(
                p_support, p_query, nsample, b_support, b_query
            )
    else:
        raise ValueError(f"Unknown grouper method: {method}")
    return grouper


class LocalAggregationPyG(nn.Module):
    """Local aggregation module using PyG operations."""

    def __init__(
        self,
        channels: List[int],
        norm_args=None,
        act_args=None,
        group_args=None,
        conv_args=None,
        feature_type="dp_fj",
        reduction="max",
        last_act=True,
        **kwargs,
    ):
        super().__init__()
        if norm_args is None:
            norm_args = {"norm": "bn1d"}
        if act_args is None:
            act_args = {"act": "relu"}
        if group_args is None:
            group_args = {"NAME": "ballquery", "radius": 0.1, "nsample": 16}
        if conv_args is None:
            conv_args = {}

        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        # Channel adjustment for feature type
        channels[0] = CHANNEL_MAP[feature_type](channels[0])

        # Build MLP layers
        convs = []
        for i in range(len(channels) - 1):
            convs.append(
                create_linearblock(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=None
                    if i == (len(channels) - 2) and not last_act
                    else act_args,
                    **conv_args,
                )
            )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)

        # Pooling operation
        reduction = "mean" if reduction.lower() == "avg" else reduction.lower()
        assert reduction in ["sum", "max", "mean"]
        self.reduction = reduction

    def forward(self, p, x, b, edge_index):
        """
        Args:
            p: (N, 3) point positions
            x: (N, C) features
            b: (N,) batch indices
            edge_index: (2, E) edge indices [source, target]
        """
        # Compute relative positions and concatenate with neighbor features
        dp = p[edge_index[1]] - p[edge_index[0]]
        xj = x[edge_index[1]]
        x_combined = torch.cat([dp, xj], dim=1)

        # Apply MLP
        x_out = self.convs(x_combined)

        # Aggregate by target nodes
        x_aggregated = scatter(
            x_out, edge_index[0], dim=0, dim_size=p.shape[0], reduce=self.reduction
        )
        return x_aggregated


class SetAbstractionPyG(nn.Module):
    """Set Abstraction module using PyG operations."""

    def __init__(
        self,
        in_channels,
        out_channels,
        layers=2,
        stride=1,
        group_args=None,
        norm_args=None,
        act_args=None,
        conv_args=None,
        sampler="fps",
        use_res=False,
        is_head=False,
    ):
        super().__init__()
        if norm_args is None:
            norm_args = {"norm": "bn1d"}
        if act_args is None:
            act_args = {"act": "relu"}
        if group_args is None:
            group_args = {"NAME": "ballquery", "radius": 0.1, "nsample": 16}
        if conv_args is None:
            conv_args = {}

        self.stride = stride
        self.is_head = is_head
        self.use_res = use_res and (stride > 1 or not is_head)

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
        channels[0] = in_channels + (
            3 if not is_head else 0
        )  # Add position dim if not head

        if self.use_res and in_channels != channels[-1]:
            self.skipconv = create_linearblock(
                in_channels, channels[-1], norm_args=None, act_args=None
            )
        elif self.use_res:
            self.skipconv = nn.Identity()

        # Build MLP
        convs = []
        for i in range(len(channels) - 1):
            convs.append(
                create_linearblock(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=None
                    if i == len(channels) - 2 and self.use_res
                    else act_args,
                    **conv_args,
                )
            )
        self.convs = nn.Sequential(*convs)
        self.act = create_act(act_args) if self.use_res else None

        if not is_head:
            self.grouper = create_grouper(group_args, support_same_as_query=False)
            if sampler.lower() == "fps":
                self.sample_fn = fps
            elif sampler.lower() == "random":
                self.sample_fn = random_sample
            else:
                self.sample_fn = fps

    def forward(self, p, x, b):
        """
        Args:
            p: (N, 3) point positions
            x: (N, C) features
            b: (N,) batch indices
        Returns:
            new_p: downsampled positions
            new_x: aggregated features
            new_b: batch indices for downsampled points
        """
        if self.is_head:
            # Head layer: just apply MLP
            x = self.convs(x)
            return p, x, b

        # Downsample points using FPS
        if self.stride > 1:
            idx = self.sample_fn(p, b, ratio=(1.0 / self.stride))
            new_p = p[idx]
            new_b = b[idx]
        else:
            idx = None
            new_p = p
            new_b = b

        # Get neighborhood
        edge_index = self.grouper(p, new_p, b, new_b)

        # Compute relative positions and neighbor features
        dp = p[edge_index[1]] - p[edge_index[0]]
        xj = x[edge_index[1]]
        x_combined = torch.cat([dp, xj], dim=1)

        # Apply MLP and aggregate
        x_out = self.convs(x_combined)
        x_aggregated = scatter(
            x_out, edge_index[0], dim=0, dim_size=new_p.shape[0], reduce="max"
        )

        if self.use_res and idx is not None:
            identity = x[idx]
            identity = self.skipconv(identity)
            x_aggregated = self.act(x_aggregated + identity)

        return new_p, x_aggregated, new_b


class FeaturePropagationPyG(nn.Module):
    """Feature Propagation module using PyG's knn_interpolate."""

    def __init__(
        self,
        mlp: List[int],
        k: int = 3,
        norm_args=None,
        act_args=None,
        **kwargs,
    ):
        super().__init__()
        if norm_args is None:
            norm_args = {"norm": "bn1d"}
        if act_args is None:
            act_args = {"act": "relu"}
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        self.k = k

        # Build MLP
        convs = []
        for i in range(len(mlp) - 1):
            convs.append(
                create_linearblock(
                    mlp[i],
                    mlp[i + 1],
                    norm_args=norm_args,
                    act_args=act_args,
                )
            )
        self.convs = nn.Sequential(*convs) if convs else nn.Identity()

    def forward(self, p_up, x_up, b_up, p_down, x_down, b_down):
        """
        Args:
            p_up: (N_up, 3) positions to upsample to
            x_up: (N_up, C_up) features to upsample to
            b_up: (N_up,) batch indices
            p_down: (N_down, 3) positions from below
            x_down: (N_down, C_down) features from below
            b_down: (N_down,) batch indices
        Returns:
            x_new: (N_up, C_new) upsampled and fused features
        """
        # Interpolate features from down to up
        x_interp = knn_interpolate(x_down, p_down, p_up, b_down, b_up, k=self.k)

        # Concatenate with existing features
        if x_up is not None:
            x_combined = torch.cat([x_up, x_interp], dim=1)
        else:
            x_combined = x_interp

        # Apply MLP
        x_new = self.convs(x_combined)
        return x_new


@MODELS.register_module()
class PointNet2EncoderPyG(nn.Module):
    """PointNet++ Encoder using PyG for variable-size point clouds.

    Args:
        in_channels: Input feature dimension
        radius: Radius for neighborhood queries
        num_samples: Number of neighbor samples
        aggr_args: Aggregation arguments
        group_args: Grouping arguments
        conv_args: Convolution arguments
        norm_args: Normalization arguments
        act_args: Activation arguments
        blocks: Number of blocks per stage
        mlps: Channel configuration per block
        width: Initial channel width
        strides: Downsampling stride per stage
        layers: Number of layers per block
        sampler: Sampling method ('fps' or 'random')
        use_res: Whether to use residual connections
    """

    def __init__(
        self,
        in_channels: int = 4,
        radius: float or List[float] = 0.1,
        num_samples: int or List[int] = 32,
        aggr_args: dict = None,
        group_args: dict = None,
        conv_args: dict = None,
        norm_args: dict = None,
        act_args: dict = None,
        blocks: Optional[List[int]] = None,
        mlps=None,
        width: Optional[int] = 32,
        strides: List[int] = [4, 4, 4, 4],
        layers: int = 2,
        sampler: str = "fps",
        use_res: bool = False,
        **kwargs,
    ):
        super().__init__()

        if aggr_args is None:
            aggr_args = {"feature_type": "dp_fj", "reduction": "max"}
        if group_args is None:
            group_args = {"NAME": "ballquery", "radius": 0.1, "nsample": 16}
        if norm_args is None:
            norm_args = {"norm": "bn1d"}
        if act_args is None:
            act_args = {"act": "relu"}
        if conv_args is None:
            conv_args = {}

        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        stages = len(strides)
        self.strides = strides
        self.blocks = blocks if blocks is not None else [1] * stages

        # Convert radius and num_samples to lists
        if isinstance(radius, (int, float)):
            radius = [radius * (2**i) for i in range(stages)]
        if isinstance(num_samples, int):
            num_samples = [num_samples] * stages

        self.radius = radius
        self.num_samples = num_samples
        logging.info(f"radius: {self.radius}, num_samples: {self.num_samples}")

        # Build SA modules
        self.SA_modules = nn.ModuleList()
        skip_channel_list = [in_channels]

        if mlps is None:
            assert width is not None
            mlps = []
            current_width = width
            for stride in strides:
                if stride > 1:
                    current_width *= 2
                mlps.append([[current_width] * layers] * 1)

        for stage_idx in range(stages):
            group_args_stage = {
                "NAME": group_args.get("NAME", "ballquery"),
                "radius": radius[stage_idx],
                "nsample": num_samples[stage_idx],
            }

            # Channel configuration
            if mlps[stage_idx]:
                channels = mlps[stage_idx][0]
            else:
                channels = [width * (2**stage_idx)] * layers

            in_ch = skip_channel_list[-1]
            out_ch = channels[-1]

            self.SA_modules.append(
                SetAbstractionPyG(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    layers=layers,
                    stride=strides[stage_idx],
                    group_args=group_args_stage,
                    norm_args=norm_args,
                    act_args=act_args,
                    conv_args=conv_args,
                    sampler=sampler,
                    use_res=use_res,
                    is_head=stage_idx == 0 and strides[stage_idx] == 1,
                )
            )
            skip_channel_list.append(out_ch)

        self.out_channels = skip_channel_list[-1]
        self.channel_list = skip_channel_list

    def forward_seg_feat(self, p, x=None, b=None):
        """Forward pass returning feature pyramids for segmentation."""
        if hasattr(p, "keys"):
            p, x, b = p["pos"], p["x"], p["batch"]

        if x is None:
            x = p.clone()
        if b is None:
            b = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)

        l_p, l_x, l_b = [p], [x], [b]

        for sa_module in self.SA_modules:
            p, x, b = sa_module(p, x, b)
            l_p.append(p)
            l_x.append(x)
            l_b.append(b)

        return l_p, l_x, l_b

    def forward(self, p, x=None, b=None):
        """Forward pass."""
        return self.forward_seg_feat(p, x, b)


@MODELS.register_module()
class PointNet2DecoderPyG(nn.Module):
    """PointNet++ Decoder using PyG for variable-size point clouds."""

    def __init__(
        self,
        encoder_channel_list: List[int],
        mlps=None,
        fp_mlps=None,
        decoder_layers: int = 1,
        k: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]

        skip_channel_list = encoder_channel_list
        fp_mlps_final = []

        if fp_mlps is None:
            # Auto-generate FP MLP channels
            # Note: When mlps is provided from encoder config, use it for consistency
            if mlps is not None:
                # First generate output channel configurations (without input size)
                fp_mlps_temp = [[mlps[0][0][0]] * (decoder_layers + 1)]
                fp_mlps_temp += [[c] * (decoder_layers + 1) for c in skip_channel_list[1:-1]]
                # Then build full MLPs with input sizes following original pattern
                for k in range(len(fp_mlps_temp)):
                    pre_channel = (
                        fp_mlps_temp[k + 1][-1] if k + 1 < len(fp_mlps_temp) else skip_channel_list[-1]
                    )
                    fp_mlps_final.append(
                        [pre_channel + skip_channel_list[k]] + fp_mlps_temp[k]
                    )
            else:
                # Fallback to auto-generation based on encoder channels
                for skip_ch in skip_channel_list[:-1]:
                    fp_mlps_final.append(
                        [self.in_channels + skip_ch] + [skip_ch] * decoder_layers
                    )
                    self.in_channels = skip_ch
        else:
            # When fp_mlps is provided, prepend the input channel size
            # Following the original PointNet2Decoder pattern
            for k in range(len(fp_mlps)):
                pre_channel = (
                    fp_mlps[k + 1][-1] if k + 1 < len(fp_mlps) else skip_channel_list[-1]
                )
                # Prepend concatenated input size to the provided fp_mlp
                fp_mlps_final.append([pre_channel + skip_channel_list[k]] + fp_mlps[k])

        self.FP_modules = nn.ModuleList()
        for fp_mlp in fp_mlps_final:
            self.FP_modules.append(FeaturePropagationPyG(fp_mlp, k=k))

        # Output channels is the final output of the first FP module
        self.out_channels = fp_mlps_final[0][-1] if fp_mlps_final else skip_channel_list[0]

    def forward(self, l_p, l_x, l_b):
        """
        Args:
            l_p: List of positions [(N_0, 3), ..., (N_k, 3)]
            l_x: List of features [(N_0, C_0), ..., (N_k, C_k)]
            l_b: List of batch indices [(N_0,), ..., (N_k,)]
        Returns:
            x: (N_0, C_out) final features at original resolution
        """
        for i in range(-1, -len(self.FP_modules) - 1, -1):
            l_x[i - 1] = self.FP_modules[i](
                l_p[i - 1], l_x[i - 1], l_b[i - 1], l_p[i], l_x[i], l_b[i]
            )

        return l_x[0]
