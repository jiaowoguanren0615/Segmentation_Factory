# Copyright (c) Meta Platforms, Inc. and affiliates.
import re

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import trunc_normal_


class HOOK(nn.Module):
    '''
    for hook fn
    '''
    def __init__(self, block_index=-1, block_name=None):
        super().__init__()
        self.block_index = block_index
        self.block_name = block_name

    def forward(self, x):
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    # split the channel, however, it is slower
    # H_split = x.split(window_size, dim = 2)
    # HW_split = []
    # for split in H_split:
    #     HW_split += split.split(window_size, dim=-1)
    # windows = torch.cat(HW_split, dim=0)
    # return windows

    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    _, C, _, _ = windows.shape
    # split the channel, however, it is slower
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # HW_split = windows.split(B, dim = 0)
    # H_split = []
    # split_size = W // window_size
    # for i in range(H // window_size):
    #     H_split.append(torch.cat(HW_split[i * split_size:(i + 1) * split_size], dim=2))
    # x = torch.cat(H_split, dim=-1)
    # return x

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x


class WindowPartion(nn.Module):
    def __init__(self,
                 window_size=0,
                 **kargs):
        super().__init__()
        assert window_size > 0
        self.window_size = window_size

    def forward(self, x):
        if type(x) is tuple:
            x = x[0]
        B, C, H, W = x.shape
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        Ho, Wo = H, W
        _, _, Hp, Wp = x.shape
        x = window_partition(x, self.window_size)
        return x, (Ho, Wo, Hp, Wp, pad_r, pad_b)


class WindowReverse(nn.Module):
    def __init__(self,
                 window_size=0,
                 **kargs):
        super().__init__()
        assert window_size > 0
        self.window_size = window_size

    def forward(self, x):
        x, (Ho, Wo, Hp, Wp, pad_r, pad_b) = x[0], x[1]
        x = window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :Ho, :Wo].contiguous()
        return x, (Ho, Wo, Hp, Wp, pad_r, pad_b)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Conv2d(torch.nn.Sequential):
    '''
    adapted from Conv2d_BN
    only test for 1x1 conv without bias
    '''
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.in_channels = a
        self.groups = groups
        self.kernel_size = ks
        self.add_module('bn', torch.nn.BatchNorm2d(a))
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, c = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[None, :, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        b = b @ c.weight.squeeze(-1).squeeze(-1).T
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepVGGDW(torch.nn.Module):
    def __init__(self,
                 dim,
                 kernel=7):
        super().__init__()
        self.conv = Conv2d_BN(dim, dim, kernel, 1, kernel//2, groups=dim)
        self.conv1 = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
        self.conv2 = torch.nn.Conv2d(dim, dim, 1, 1, 0, groups=dim)
        self.dim = dim
        self.bn = torch.nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x) + self.conv2(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        conv2 = self.conv2

        conv_w = conv.weight
        conv_b = conv.bias

        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        conv2_w = conv2.weight
        conv2_b = conv2.bias
        conv2_w = torch.nn.functional.pad(conv2_w, [3, 3, 3, 3])
        identity = torch.nn.functional.pad(torch.ones(conv2_w.shape[0], conv2_w.shape[1], 1, 1, device=conv2_w.device),
                                           [3, 3, 3, 3])

        final_conv_w = conv_w + conv1_w + conv2_w + identity
        final_conv_b = conv_b + conv1_b + conv2_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop_path=0., layer_scale_init_value=0, dim=None):
        super().__init__()
        self.m = m
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                      requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x):
        if self.gamma is not None:
            return x + self.gamma * self.drop_path(self.m(x))
        else:
            return x + self.drop_path(self.m(x))

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            kernel_size = m.kernel_size[0] // 2
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [kernel_size, kernel_size, kernel_size, kernel_size])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            kernel_size = m.kernel_size[0] // 2
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [kernel_size, kernel_size, kernel_size, kernel_size])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class SHMA(nn.Module):
    fused_attn: torch.jit.Final[bool]
    def __init__(
            self,
            dim,
            num_heads=1,
            attn_drop=0.,
            fused_attn=False,
            ratio=4,
            q_kernel=1,
            kv_kernel=1,
            kv_stride=1,
            head_dim_reduce_ratio=4,
            window_size=0,
            sep_v_gate=False,
            **kwargs,
    ):
        super().__init__()
        mid_dim = int(dim * ratio)
        dim_attn = dim // head_dim_reduce_ratio
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.v_dim_head = mid_dim // self.num_heads
        self.scale = self.dim_head ** -0.5
        self.fused_attn = fused_attn

        self.q = Conv2d_BN(dim, dim_attn, q_kernel, stride=1, pad=q_kernel // 2)
        self.k = Conv2d_BN(dim, dim_attn, kv_kernel, stride=kv_stride, pad=kv_kernel // 2)
        self.gate_act = nn.Sigmoid()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Conv2d_BN(mid_dim, dim, 1)
        self.window_size = window_size
        self.block_index = kwargs['block_index']
        self.kv_stride = kv_stride
        self.sep_v_gate = sep_v_gate
        self.v_gate = Conv2d_BN(dim, 2 * mid_dim, kv_kernel, stride=kv_stride, pad=kv_kernel // 2)

    def forward(self, x, attn_mask=None):
        B, C, H, W = x.shape
        if self.window_size:
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0))
            Ho, Wo = H, W
            _, _, Hp, Wp= x.shape
            x = window_partition(x, self.window_size)
            B, C, H, W = x.shape
        v, gate = self.gate_act(self.v_gate(x)).chunk(2, dim=1)
        q_short = self.q(x)
        q = q_short.flatten(2)
        k = self.k(x).flatten(2)

        v = v.flatten(2)
        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2).contiguous(),
                k.transpose(-1, -2).contiguous(),
                v.transpose(-1, -2).contiguous(),
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ).transpose(-1, -2).reshape(B, -1, H, W)
        else:
            q = q * self.scale
            attn = q.transpose(-2, -1) @ k
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)

        x = x * gate
        x = self.proj(x)
        if self.window_size:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :Ho, :Wo].contiguous()
        return x


class SHMABlock(nn.Module):
    def __init__(self,
                 window_split=False,
                 window_reverse=False,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 **kargs,
        ):
        super().__init__()
        self.window_split=window_split
        self.window_reverse = window_reverse
        if self.window_split or self.window_reverse:
            self.window_size = kargs['window_size']
            kargs['window_size'] = 0
        self.token_channel_mixer = Residual(
                SHMA(**kargs),
                drop_path=drop_path,
                layer_scale_init_value=layer_scale_init_value,
                dim=kargs['dim'],
            )

    def forward(self, x):
        if self.window_split:
            if type(x) is tuple:
                x = x[0]
            B, C, H, W = x.shape
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0))
            Ho, Wo = H, W
            _, _, Hp, Wp = x.shape
            # x = window_partition(x, self.window_size)
            x_split = x.chunk(16, dim=1)
            new_x = []
            for split in x_split:
                new_x.append(window_partition(split, self.window_size))
            x = torch.cat(new_x, dim=1)
            x = (x, (Ho, Wo, Hp, Wp, pad_r, pad_b))
        if type(x) is tuple:
            result = self.token_channel_mixer(x[0]), x[1]
        else:
            result = self.token_channel_mixer(x)
        if self.window_reverse:
            x, (Ho, Wo, Hp, Wp, pad_r, pad_b) = result[0], result[1]
            # x = window_reverse(x, self.window_size, Hp, Wp)
            x_split = x.chunk(16, dim=1)
            new_x = []
            for split in x_split:
                new_x.append(window_reverse(split, self.window_size, Hp, Wp))
            x = torch.cat(new_x, dim=1)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :Ho, :Wo].contiguous()
            return x, (Ho, Wo, Hp, Wp, pad_r, pad_b)
        else:
            return result


class FFN2d(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 ratio=4,
                 act_layer=nn.GELU,
                 **kargs):
        super().__init__()
        mid_chs = ratio * dim
        self.channel_mixer = Residual(nn.Sequential(
            Conv2d_BN(dim, mid_chs),
            act_layer(),
            Conv2d_BN(mid_chs, dim)),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=dim,
        )

    def forward(self, x):
        if type(x) is tuple:
            return self.channel_mixer(x[0]), x[1]
        else:
            return self.channel_mixer(x)


class ConvBlock(nn.Module):
    def __init__(self,
                 dim,
                 out_dim=None,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 kernel=7,
                 stride=1,
                 ratio=4,
                 act_layer=nn.GELU,
                 reparameterize=False,
                 **kargs,
        ):
        super().__init__()
        mid_chs = ratio * dim
        if out_dim is None:
            out_dim = dim
        if reparameterize:
            assert stride == 1
            dw_conv = eval(reparameterize)(dim, kernel)
        else:
            dw_conv = Conv2d_BN(dim, dim, kernel, stride, pad=kernel // 2, groups=dim)  # depthwise conv
        self.token_channel_mixer = Residual(nn.Sequential(
            dw_conv,
            Conv2d_BN(dim, mid_chs),
            act_layer(),
            Conv2d_BN(mid_chs, out_dim)),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=out_dim,
        )

    def forward(self, x):
        if type(x) is tuple:
            return self.token_channel_mixer(x[0]), x[1]
        else:
            return self.token_channel_mixer(x)


class RepCPE(nn.Module):
    def __init__(self,
                 dim,
                 kernel=7,
                 **kargs,
        ):
        super().__init__()
        self.cpe = Residual(
            Conv2d_BN(dim, dim, kernel, 1, pad=kernel//2, groups=dim)
        )
    def forward(self, x):
        if type(x) is tuple:
            return self.cpe(x[0]), x[1]
        else:
            return self.cpe(x)


class BasicBlock(nn.Module):
    """
    parse the block_type and arguments
    refer to timm.models._efficientnet_builder._decode_block_str, thanks
    """
    def __init__(self,
                 dim,
                 out_dim=None,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 block_type=None,
                 block_index=-1,
        ):
        super().__init__()
        args = block_type.split('_')
        args_dict = {
            "dim": dim,
            "out_dim": out_dim,
            "drop_path": drop_path,
            "layer_scale_init_value": layer_scale_init_value,
            "block_index": block_index
        }
        block = args[0]
        for arg in args[1:]:
            splits = re.split(r'(\d.*)', arg)
            key, value = splits[:2]
            if key == 'k':
                v = int(arg[1:])
                args_dict['kernel'] = v
            elif key == 'qk':
                v = int(arg[2:])
                args_dict['q_kernel'] = v
            elif key == 'reparam':
                v = int(arg[7:])
                if v == 1:
                    args_dict['reparameterize'] = 'RepVGGDW'
            elif key == 'kvk':
                v = int(arg[3:])
                args_dict['kv_kernel'] = v
            elif key == 'id':
                v = int(arg[2:])
                args_dict['dim'] = v
            elif key == 'od':
                v = int(arg[2:])
                args_dict['out_dim'] = v
            elif key == 's':
                v = int(arg[1:])
                args_dict['stride'] = v
            elif key == 'kvs':
                v = int(arg[3:])
                args_dict['kv_stride'] = v
            elif key == 'hdrr':
                v = int(arg[4:])
                args_dict['head_dim_reduce_ratio'] = v
            elif key == 'nh':
                v = int(arg[2:])
                args_dict['num_heads'] = v
            elif key == 'ek':
                v = int(arg[2:])
                args_dict['extraDW_kernel'] = v
            elif key == 'r':
                v = int(arg[1:])
                args_dict['ratio'] = v
            elif key == 'ws':
                v = int(arg[2:])
                args_dict['window_size'] = v
            elif key == 'wsp':
                v = int(arg[3:])
                args_dict['window_split'] = v
            elif key == 'wre':
                v = int(arg[3:])
                args_dict['window_reverse'] = v
            elif key == 'fa':
                v = int(arg[2:])
                args_dict['fused_attn'] = v
            elif key == 'svg':
                v = int(arg[3:])
                args_dict['sep_v_gate'] = v
            elif key == 'ds':
                v = int(arg[2:])
                args_dict['downsample'] = v
            elif key == 'act':
                v = int(arg[3:])
                if v == 0:
                    args_dict['act_layer'] = nn.Identity
                elif v == 1:
                    args_dict['act_layer'] = nn.ReLU
                elif v == 2:
                    args_dict['act_layer'] = nn.GELU
                elif v == 3:
                    args_dict['act_layer'] = nn.Hardswish
            elif key == 'norm':
                v = int(arg[4:])
                if v == 0:
                    args_dict['norm_layer'] = nn.Identity
                elif v == 1:
                    args_dict['norm_layer'] = nn.LayerNorm
        self.block = eval(block)(**args_dict)

    def forward(self, x):
        return self.block(x)


class EdgeResidual(nn.Module):
    """ FusedIB in Miblenetv4_conv_medium from timm, thanks
    """
    def __init__(self,
                 in_chs: int,
                 out_chs: int,
                 exp_kernel_size=3,
                 stride=1,
                 exp_ratio=1.0,
                 act_layer=nn.ReLU,
                 ):
        super(EdgeResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.conv_exp_bn1 = Conv2d_BN(in_chs, mid_chs, exp_kernel_size, stride, pad=exp_kernel_size//2)
        self.act = act_layer()
        self.conv_pwl_bn2 = Conv2d_BN(mid_chs, out_chs, 1)

    def forward(self, x):
        x = self.conv_exp_bn1(x)
        x = self.act(x)
        x = self.conv_pwl_bn2(x)
        return x


class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=False):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class iFormer(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        conv_stem_type (str): choose which convolutional stem to use.
        use_bn (bool): whether to use BN for all projection.
        distillation (bool): whether to use knowledge distillation as DeiT.
        last_proj (bool): whether to add another projection before classify head as MobileNetV3.
        sep_downsample (bool): whether to use seperate downsample layers.
        block_types (str): choose which block to use.
        act_layer: activation layer.
        downsample_kernels: the convolution kernel for downsample layers, the first is for stem.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=0, head_init_scale=1.,
                 conv_stem_type='FusedIB',
                 use_bn=False,
                 distillation=False,
                 last_proj=False,
                 sep_downsample=True,
                 block_types=None,
                 act_layer=nn.GELU,
                 downsample_kernels=[3, 3, 3, 3],
                 **kwargs
                 ):
        super().__init__()
        self.use_bn = use_bn
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_kernel = downsample_kernels[0]
        if conv_stem_type == 'FusedIB':
            stem = nn.Sequential(Conv2d_BN(3, dims[0] // 2, stem_kernel, 2, stem_kernel//2),
                                 act_layer(),
                                 EdgeResidual(dims[0] // 2, dims[0], stem_kernel, 2, exp_ratio=4, act_layer=act_layer)
                                )
        elif conv_stem_type == 'conv_stem':
            stem = nn.Sequential(Conv2d_BN(3, dims[0] // 2, stem_kernel, 2, pad=stem_kernel//2),
                                 act_layer(),
                                 Conv2d_BN(dims[0] // 2, dims[0], stem_kernel, 2, pad=stem_kernel//2)
                                )
        elif conv_stem_type == 'ConvNeXt':
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        else:
            raise  NotImplementedError('do not support now!')
        self.downsample_layers.append(stem)
        self.sep_downsample = sep_downsample
        for i in range(3):
            if not self.sep_downsample:
                downsample_layer = nn.Identity()
            elif use_bn:
                downsample_layer = nn.Sequential(
                    Conv2d_BN(dims[i], dims[i + 1], downsample_kernels[i+1], 2, pad=downsample_kernels[i+1]//2)
                )
            else:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BasicBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        block_type=block_types[cur + j],
                        block_index=cur+j) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.last_proj = last_proj
        if use_bn:
            cur_dim = dims[-1]
            if last_proj:
                self.proj = BN_Linear(cur_dim, cur_dim*2)
                self.act = act_layer()
                cur_dim = cur_dim * 2
            self.classifier = Classfier(cur_dim, num_classes, distillation)
        else:
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_intermediate(self, x):
        features = []
        for i in range(4):
            if type(x) is tuple:
                x, others = x
                x = (self.downsample_layers[i](x), others)
            else:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward_features(self, x):
        for i in range(4):
            if type(x) is tuple:
                x, others = x
                x = (self.downsample_layers[i](x), others)
            else:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        if type(x) is tuple:
            x = x[0]
        if self.use_bn:
            x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            if self.last_proj:
                x = self.proj(x)
                x = self.act(x)
            return self.classifier(x)
        else:
            x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
            return self.head(x)

    def forward(self, x):
        x = self.forward_intermediate(x)
        return x

@register_model
def iFormer_t(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [2, 2, 16, 6]
    block_types = ['ConvBlock_k7_r3'] * 2 + ['ConvBlock_k7_r3'] * 2 + ['ConvBlock_k7_r3'] * 6 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1', 'FFN2d_r2'] * 3 + ['ConvBlock_k7_r3'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1', 'FFN2d_r2'] * 2
    model = iFormer(depths=depths, dims=[32, 64, 128, 256], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

@register_model
def iFormer_s(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [2, 2, 19, 6]
    block_types = ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 9 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1', 'FFN2d_r3'] * 3 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1', 'FFN2d_r3'] * 2
    model = iFormer(depths=depths, dims=[32, 64, 176, 320], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

@register_model
def iFormer_m(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [2, 2, 22, 6]
    block_types = ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 9 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1', 'FFN2d_r3'] * 4 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1', 'FFN2d_r3'] * 2
    model = iFormer(depths=depths, dims=[48, 96, 192, 384], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

"""
All models named faster are only used for latency measurement of detection and segmentation here.
"""
@register_model
def iFormer_m_faster(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [2, 2, 22, 6]
    block_types = ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 9 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wsp1_fa1', 'FFN2d_r3'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_fa1', 'FFN2d_r3'] * 2 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wre1_fa1', 'FFN2d_r3'] + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1_fa1', 'FFN2d_r3'] * 2
    model = iFormer(depths=depths, dims=[48, 96, 192, 384], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

@register_model
def iFormer_l(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [2, 2, 33, 6]
    block_types = ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 8 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1', 'FFN2d_r3'] * 8 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1', 'FFN2d_r3'] * 2
    model = iFormer(depths=depths, dims=[48, 96, 256, 384], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

@register_model
def iFormer_l_faster(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [2, 2, 33, 6]
    block_types = ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 2 + ['ConvBlock_k7_r4'] * 8 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wsp1_fa1', 'FFN2d_r3'] * 1 +\
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_fa1', 'FFN2d_r3'] * 5 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wre1_fa1', 'FFN2d_r3'] * 1 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_fa1', 'FFN2d_r3'] * 1 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1_fa1', 'FFN2d_r3'] * 2
    model = iFormer(depths=depths, dims=[48, 96, 256, 384], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

@register_model
def iFormer_l2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [3, 3, 46, 9]
    block_types = ['ConvBlock_k7_r4'] * 3 + ['ConvBlock_k7_r4'] * 3 + ['ConvBlock_k7_r4'] * 12 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1', 'FFN2d_r3'] * 11 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1', 'FFN2d_r3'] * 3
    model = iFormer(depths=depths, dims=[64, 128, 256, 512], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

@register_model
def iFormer_l2_faster(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [3, 3, 46, 9]
    block_types = ['ConvBlock_k7_r4'] * 3 + ['ConvBlock_k7_r4'] * 3 + ['ConvBlock_k7_r4'] * 12 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wsp1_fa1', 'FFN2d_r3'] * 1 +\
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_fa1', 'FFN2d_r3'] * 9 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_ws16_wre1_fa1', 'FFN2d_r3'] * 1 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr2_act0_nh1_fa1', 'FFN2d_r3'] * 1 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr4_act0_nh1_fa1', 'FFN2d_r3'] * 3
    model = iFormer(depths=depths, dims=[48, 128, 256, 448], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types, **kwargs)
    return model

@register_model
def iFormer_h(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    depths = [5, 5, 60, 18]
    block_types = ['ConvBlock_k7_r4'] * 5 + ['ConvBlock_k7_r4'] * 5 + ['ConvBlock_k7_r4'] * 14 + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr1_act0_nh1', 'FFN2d_r4'] * 15 + ['ConvBlock_k7_r4'] + \
                  ['RepCPE_k3', 'SHMABlock_r1_hdrr1_act0_nh1', 'FFN2d_r4'] * 6
    model = iFormer(depths=depths, dims=[96, 192, 384, 768], use_bn=True, conv_stem_type='FusedIB',
                     downsample_kernels=[5, 3, 3, 3],
                     block_types=block_types,
                     **kwargs)
    return model

# if __name__ == '__main__':
#     net = iFormer_m(num_classes=5)
#     x = torch.randn(1, 3, 224, 224)
#     y = net(x)
#     for i in y:
#         print(i.shape)