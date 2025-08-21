# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_plugin_layer
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS

from einops import rearrange
from functools import partial
import math
from typing import Iterable
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, trunc_normal_

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, constant_init, normal_init
from mmcv.cnn import ConvModule
from torch import Tensor
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmdet.utils import MultiConfig, OptConfigType, OptMultiConfig
from mmengine.utils import digit_version, is_tuple_of
from timm.models.layers import to_2tuple, trunc_normal_
from torchvision.utils import save_image

from mmyolo.registry import MODELS

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import transforms


#region GGMix
def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='border',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=x.dtype),
        torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.data_format = data_format
        
        normalized_dim = len(self.normalized_shape)
        param_shape = self.normalized_shape
        
        # 初始化gamma为1.0，beta为0.0
        self.gamma = nn.Parameter(torch.ones(*param_shape) * 1.0)
        self.beta = nn.Parameter(torch.zeros(*param_shape))
        
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {self.data_format}")
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # 使用F.layer_norm进行channels_last层归一化
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        
        # 自定义实现channels_first
        channel_dim = 1
        
        # 计算通道维度的均值
        mean = x.mean(dim=channel_dim, keepdim=True)
        
        # 计算通道维度的方差，使用epsilon进行数值稳定性
        var = x.var(dim=channel_dim, keepdim=True, unbiased=False)
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        reshaped_gamma = self.gamma.view((-1,) + (1,) * (x.ndim - 2))
        reshaped_beta = self.beta.view((-1,) + (1,) * (x.ndim - 2))
        
        return x_normalized * reshaped_gamma + reshaped_beta



class Global_Guidance(nn.Module):
    def __init__(self, dim, window_size=4, k=4,ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2
        
        # 输入卷积层
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # 输出偏移量
        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        # 输出mask
        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        # 输出通道注意力
        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

        # 输出空间注意力
        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        


    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):
        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)
        
        x = torch.mean(x, keepdim=True, dim=1) 

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        return mask, offsets, ca, sa


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, N, C = q.shape
        #由外部输入的q, k, v进行线性变换，并reshape为(B, N, num_heads, head_dim)，然后permute为(num_heads, B, N, head_dim)
        q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        k = self.k_proj(k).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        v = self.v_proj(v).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bottleneck_ratio=0.5, groups=4):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步幅
        :param padding: 填充
        :param bottleneck_ratio: 用于瓶颈层的通道缩减比例
        :param groups: 逐点卷积的分组数
        """
        super().__init__()
        bottleneck_channels = int(in_channels * bottleneck_ratio)
        
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        
        # 瓶颈层降维
        self.bottleneck = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        
        # 分组逐点卷积
        self.pointwise = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bottleneck(x)
        x = self.pointwise(x)
        return x
    

class GGmix(nn.Module):
    def __init__(self, dim, window_size=4, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.project_q = nn.Linear(dim, dim, bias=bias)
        self.project_k = nn.Linear(dim, dim, bias=bias)

        self.multihead_attn = MultiHeadAttention(dim)
        self.conv_sptial = DepthwiseSeparableConv(dim, dim)   
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)

        self.act = nn.GELU()
        self.route = Global_Guidance(dim, window_size, ratio=ratio)

        # 生成global feature
        self.global_predictor = nn.Sequential(
            nn.Conv2d(3, 8, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(8, dim+2, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, condition_global=None, mask=None, train_mode=True, img_ori=None):
        N, C, H, W = x.shape

        global_status = self.global_predictor(img_ori)
        global_status = F.interpolate(global_status, size=(H, W), mode='bilinear', align_corners=False)

        if self.is_deformable:
            patch_status = torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.window_size), torch.linspace(-1, 1, self.window_size)))\
                .type_as(x).unsqueeze(0).repeat(N, 1, H // self.window_size, W // self.window_size)
            
            global_feature = torch.cat([global_status, patch_status], dim=1)
            
        mask, offsets, ca, sa = self.route(global_feature, ratio=self.ratio, train_mode=train_mode)

        q = x
        k = x + flow_warp(x, offsets.permute(0, 2, 3, 1))
        qk = torch.cat([q, k], dim=1)
        v = self.project_v(x)

        vs = v * sa

        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        N_ = v.shape[1]
        v1, v2 = v * mask, vs * (1 - mask)
        qk1 = qk * mask
    

        v1 = rearrange(v1, 'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1, 'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        # 使用多头自注意力机制
        f_attn = self.multihead_attn(q1, k1, v1)

        f_attn = rearrange(f_attn, '(b n) (dh dw) c -> b n (dh dw c)',
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        
        attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H // self.window_size, w=W // self.window_size, dh=self.window_size, dw=self.window_size
        )

        out = attn_out
        out = self.act(self.conv_sptial(out)) * ca + out
        out = self.project_out(out)

        return out

#endregion


@MODELS.register_module()
class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """BaseBackbone backbone used in YOLO series.

    .. code:: text

     Backbone model structure diagram
     +-----------+
     |   input   |
     +-----------+
           v
     +-----------+
     |   stem    |
     |   layer   |
     +-----------+
           v
     +-----------+
     |   stage   |
     |  layer 1  |
     +-----------+
           v
     +-----------+
     |   stage   |
     |  layer 2  |
     +-----------+
           v
         ......
           v
     +-----------+
     |   stage   |
     |  layer n  |
     +-----------+
     In P5 model, n=4
     In P6 model, n=5

    Args:
        arch_setting (list): Architecture of BaseBackbone.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels: Number of input image channels. Defaults to 3.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to None.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch_setting: list,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('"frozen_stages" must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.plugins = plugins

        self.stem = self.build_stem_layer()
        self.layers = ['stem']

        for idx, setting in enumerate(arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            if plugins is not None:
                stage += self.make_stage_plugins(plugins, idx, setting)
            self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx + 1}')

        # yolov8x
        self.GGmix0 = GGmix(128).cuda()
        self.GGmix1 = GGmix(256).cuda()
        self.GGmix2 = GGmix(512).cuda()
        self.ln0 = LayerNorm(128).cuda()
        self.ln1 = LayerNorm(256).cuda()
        self.ln2 = LayerNorm(512).cuda()


    @abstractmethod
    def build_stem_layer(self):
        """Build a stem layer."""
        pass

    @abstractmethod
    def build_stage_layer(self, stage_idx: int, setting: list):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        pass

    def make_stage_plugins(self, plugins, stage_idx, setting):
        """Make plugins for backbone ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block``, ``dropout_block``
        into the backbone.


        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True)),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True)),
            ... ]
            >>> model = YOLOv5CSPDarknet()
            >>> stage_plugins = model.make_stage_plugins(plugins, 0, setting)
            >>> assert len(stage_plugins) == 1

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1 -> conv2 -> conv3 -> yyy

        Suppose ``stage_idx=1``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1 -> conv2 -> conv3 -> xxx -> yyy


        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build
                If stages is missing, the plugin would be applied to all
                stages.
            setting (list): The architecture setting of a stage layer.

        Returns:
            list[nn.Module]: Plugins for current stage
        """
        # TODO: It is not general enough to support any channel and needs
        # to be refactored
        in_channels = int(setting[1] * self.widen_factor)
        plugin_layers = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                name, layer = build_plugin_layer(
                    plugin['cfg'], in_channels=in_channels)
                plugin_layers.append(layer)
        return plugin_layers

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        x_ori = x
        img_ori = x
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        # with residual connection
        mid1 = self.GGmix1(outs[1], img_ori=img_ori)
        outs[1] = self.ln1(outs[1] + mid1[0])
    
        mid0 = self.GGmix0(outs[0], img_ori=img_ori)
        outs[0] = self.ln0(outs[0] + mid0[0])

        mid2 = self.GGmix2(outs[2], img_ori=img_ori)
        outs[2] = self.ln2(outs[2] + mid2[0])

        outs.append(x_ori)

        return tuple(outs)
