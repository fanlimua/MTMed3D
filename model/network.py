from __future__ import annotations
from collections.abc import Sequence
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
import monai
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option
# from model.backbone import SwinTransformer, PatchMerging, PatchMergingV2
from model.backbone import SwinTransformer, PatchMerging, PatchMergingV2
from collections.abc import Callable
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Dict
import torch

from monai.networks.layers.factories import Conv, Pool

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

class SegmentationDecoder(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            # >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            #
            # # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            # >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            #
            # # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            # >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        # window_size = ensure_tuple_rep(4, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        # self.swinViT = SwinTransformer(
        #     in_chans=in_channels,
        #     embed_dim=feature_size,
        #     window_size=window_size,
        #     patch_size=patch_size,
        #     depths=depths,
        #     num_heads=num_heads,
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     drop_rate=drop_rate,
        #     attn_drop_rate=attn_drop_rate,
        #     drop_path_rate=dropout_path_rate,
        #     norm_layer=nn.LayerNorm,
        #     use_checkpoint=use_checkpoint,
        #     spatial_dims=spatial_dims,
        #     downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        #     use_v2=use_v2,
        # )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, hidden_states_out, x_in):
        # Require both input image and outputs of encoder
        # hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits 


# Swin UNETR + DenseNet121
class ClassificationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=3, out_channels=2) #Use the whole Swin UNETR
        
    def forward(self, x):
        x = self.head(x)
        return x

class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        spatial_dims: 2D or 3D images
        in_channels_list: number of channels for each feature map that
            is passed to the module
        out_channels: number of channels of the FPN representation
        extra_blocks: if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = FeaturePyramidNetwork(2, [10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels_list: list[int],
        out_channels: int,
        extra_blocks: ExtraFPNBlock | None = None,
    ):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = conv_type(in_channels, out_channels, 1)
            layer_block_module = conv_type(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        conv_type_: type[nn.Module] = Conv[Conv.CONV, spatial_dims]
        for m in self.modules():
            if isinstance(m, conv_type_):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0.0)
                
        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise AssertionError
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    # def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x: feature maps for each feature level.

        Returns:
            feature maps after FPN layers. They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        # names = list(x.keys())
        # x_values: list[Tensor] = list(x.values())

        # ------------FPN-----------------
        # last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # results = []
        # results.append(self.get_result_from_layer_blocks(last_inner, -1))

        # for idx in range(len(x) - 2, -1, -1):
        #     inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
        #     feat_shape = inner_lateral.shape[2:]
        #     inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
        #     last_inner = inner_lateral + inner_top_down
        #     results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # # -------------PANet-------------
        # Top-Down Pathway (FPN)
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # Bottom-Up Path Aggregation - Use the FPN output feature maps as inputs
        bottom_up_features = results  
        for idx in range(1, len(bottom_up_features)):
            lower_shape = bottom_up_features[idx].shape[2:]
            inner_bottom_up = F.interpolate(bottom_up_features[idx - 1], size=lower_shape, mode="nearest")
            #Rewrite the bottom_up_features except idx = 0, direct copy the feature idx = 0
            bottom_up_features[idx] = bottom_up_features[idx] + inner_bottom_up

        # Final results
        final_results = []
        for idx in range(len(bottom_up_features)):
            final_results.append(self.get_result_from_layer_blocks(bottom_up_features[idx], idx))

        return final_results
        # # -------------PANet-------------
            
        # if self.extra_blocks is not None:
        #     results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        # out = OrderedDict(list(zip(names, results)))

        # return results

class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Same code as https://github.com/pytorch/vision/blob/release/0.12/torchvision/ops/feature_pyramid_network.py
    """

    def forward(self, results: list[Tensor], x: list[Tensor], names: list[str]):
        """
        Compute extended set of results of the FPN and their names.

        Args:
            results: the result of the FPN
            x: the original feature maps
            names: the names for each one of the original feature maps

        Returns:
            - the extended set of results of the FPN
            - the extended set of names for the results
        """
        pass

class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d or max_pool3d on top of the last feature map. Serves as an ``extra_blocks``
    in :class:`~monai.networks.blocks.feature_pyramid_network.FeaturePyramidNetwork` .
    """

    def __init__(self, spatial_dims: int):
        super().__init__()
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        self.maxpool = pool_type(kernel_size=1, stride=2, padding=0)

    def forward(self, results: list[Tensor], x: list[Tensor], names: list[str]) -> tuple[list[Tensor], list[str]]:
        names.append("pool")
        results.append(self.maxpool(results[-1]))
        return results, names

class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    This head takes a list of feature maps as inputs, and outputs a list of classification maps.
    Each output map has same spatial size with the corresponding input feature map,
    and the number of output channel is num_anchors * num_classes.

    Args:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
        num_classes: number of classes to be predicted
        spatial_dims: spatial dimension of the network, should be 2 or 3.
        prior_probability: prior probability to initialize classification convolutional layers.
    """

    def __init__(
        self, in_channels: int, num_anchors: int, num_classes: int, spatial_dims: int, prior_probability: float = 0.01
    ):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        conv = []
        for _ in range(4):
            conv.append(conv_type(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(num_groups=8, num_channels=in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, conv_type):  # type: ignore
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = conv_type(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        """
        It takes a list of feature maps as inputs, and outputs a list of classification maps.
        Each output classification map has same spatial size with the corresponding input feature map,
        and the number of output channel is num_anchors * num_classes.

        Args:
            x: list of feature map, x[i] is a (B, in_channels, H_i, W_i) or (B, in_channels, H_i, W_i, D_i) Tensor.

        Return:
            cls_logits_maps, list of classification map. cls_logits_maps[i] is a
            (B, num_anchors * num_classes, H_i, W_i) or (B, num_anchors * num_classes, H_i, W_i, D_i) Tensor.

        """
        cls_logits_maps = []

        if isinstance(x, Tensor):
            feature_maps = [x]
        else:
            feature_maps = x

        for features in feature_maps:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            cls_logits_maps.append(cls_logits)

            if torch.isnan(cls_logits).any() or torch.isinf(cls_logits).any():
                if torch.is_grad_enabled():
                    raise ValueError("cls_logits is NaN or Inf.")
                else:
                    warnings.warn("cls_logits is NaN or Inf.")

        return cls_logits_maps
    
class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    This head takes a list of feature maps as inputs, and outputs a list of box regression maps.
    Each output box regression map has same spatial size with the corresponding input feature map,
    and the number of output channel is num_anchors * 2 * spatial_dims.

    Args:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
        spatial_dims: spatial dimension of the network, should be 2 or 3.
    """

    def __init__(self, in_channels: int, num_anchors: int, spatial_dims: int):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        conv = []
        for _ in range(4):
            conv.append(conv_type(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(num_groups=8, num_channels=in_channels))
            conv.append(nn.ReLU())

        self.conv = nn.Sequential(*conv)

        self.bbox_reg = conv_type(in_channels, num_anchors * 2 * spatial_dims, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, conv_type):  # type: ignore
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        """
        It takes a list of feature maps as inputs, and outputs a list of box regression maps.
        Each output box regression map has same spatial size with the corresponding input feature map,
        and the number of output channel is num_anchors * 2 * spatial_dims.

        Args:
            x: list of feature map, x[i] is a (B, in_channels, H_i, W_i) or (B, in_channels, H_i, W_i, D_i) Tensor.

        Return:
            box_regression_maps, list of box regression map. cls_logits_maps[i] is a
            (B, num_anchors * 2 * spatial_dims, H_i, W_i) or (B, num_anchors * 2 * spatial_dims, H_i, W_i, D_i) Tensor.

        """
        box_regression_maps = []

        if isinstance(x, Tensor):
            feature_maps = [x]
        else:
            feature_maps = x

        for features in feature_maps:
            box_regression = self.conv(features)
            box_regression = self.bbox_reg(box_regression)

            box_regression_maps.append(box_regression)

            if torch.isnan(box_regression).any() or torch.isinf(box_regression).any():
                if torch.is_grad_enabled():
                    raise ValueError("box_regression is NaN or Inf.")
                else:
                    warnings.warn("box_regression is NaN or Inf.")

        return box_regression_maps

class DetectionDecoder(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int]= [96,192,384,768],
        out_channels: int = 96,
        spatial_dims: int = 3,
        num_anchors: int = 5,
        num_classes: int = 1,
        feature_map_channels: int = 96,
    ) -> None:
        super().__init__()
        extra_blocks = LastLevelMaxPool(spatial_dims)
        
        self.fpn = FeaturePyramidNetwork(
                spatial_dims=spatial_dims,
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
            )
        self.num_anchors = num_anchors
        self.classification_head = RetinaNetClassificationHead(
            feature_map_channels, self.num_anchors, num_classes, spatial_dims=spatial_dims
        )
        self.regression_head = RetinaNetRegressionHead(
            feature_map_channels, self.num_anchors, spatial_dims=spatial_dims
        )
           
    def forward(self, hidden_states_out):
        hidden_states_out = hidden_states_out[-4:]
        features: List[Tensor] = self.fpn(hidden_states_out)  # FPN
        if isinstance(features, Tensor):
            feature_maps = [features]
        elif torch.jit.isinstance(features, Dict[str, Tensor]):
            feature_maps = list(features.values())
        else:
            feature_maps = list(features)

        if not isinstance(feature_maps[0], Tensor):
            raise ValueError("feature_extractor output format must be Tensor, Dict[str, Tensor], or Sequence[Tensor].")

        # compute classification and box regression maps from the feature maps
        # expandable for mask prediction in the future
        head_outputs = {"classification": self.classification_head(feature_maps)}
        head_outputs["box_regression"] = self.regression_head(feature_maps)
        return head_outputs
    
class MultiSwin(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
        )
        self.seg_decoder = SegmentationDecoder(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=use_checkpoint,
        )
        self.cls_decoder = ClassificationDecoder()
        self.det_decoder = DetectionDecoder()

    def forward(self, x, task=None):
        # Swin Transformer based Encoder
        shared_features = self.encoder(x)

        # Single Task
        if task == "segmentation":
            return self.seg_decoder(shared_features, x)
        elif task == "classification":
            seg_output = self.seg_decoder(shared_features, x)
            return self.cls_decoder(seg_output)
        elif task == "detection":
            return self.det_decoder(shared_features)
        # Multi Task
        else:
            seg_output = self.seg_decoder(shared_features, x)
            cls_output = self.cls_decoder(seg_output)
            det_output = self.det_decoder(shared_features)
            return seg_output, cls_output, det_output


# def save_model_structure(model, file_path):
#     with open(file_path, 'w') as f:
#         f.write(str(model))

# model = MultiSwin(img_size=(96,96,96), in_channels=4)
# save_model_structure(model, "multi_task_model.txt")
# print("Model structure saved to multi_task_model.txt")