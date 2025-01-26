import math
from typing import Tuple, Union

import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model. See `End-to-End Object Detection
    with Transformers <https://arxiv.org/pdf/2005.12872>`_ for more details.

    :param num_pos_feats: The feature dimension for each position along x-axis or y-axis.
        The final returned dimension for each position is 2 times of the input value,
        defaults to 64
    :param temperature: The temperature used for scaling the position embedding, defaults to 10000
    :param normalize: Whether to normalize the position embedding, defaults to False
    :param scale: A scale factor that scales the position embedding, which is used only when
        `normalize` is True, defaults to 2*math.pi
    :param eps: A value added to the denominator for numerical stability, defaults to 1e-6
    :param offset: An offset added to embed, defaults to 0.0
    """
    def __init__(
        self,
        num_pos_feats=64,
        temperature: Union[int, Tuple[int, int]] = 10000,
        normalize=False,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
    ):
        super().__init__()
        dim_t = 2 * torch.arange(num_pos_feats).div(2, rounding_mode="floor") / num_pos_feats
        if isinstance(temperature, int):
            dim_tx = dim_ty = temperature**dim_t
        else:
            assert len(temperature) == 2, "Only support two elements as (t_x, t_y) in temperature"
            dim_tx, dim_ty = [t**dim_t for t in temperature]
        self.register_buffer("dim_tx", dim_tx)
        self.register_buffer("dim_ty", dim_ty)

        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor):
        mask = mask.to(torch.int)
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        else:
            # RT-DETR uses unnormalized encoding with index from 0
            y_embed = y_embed + self.offset
            x_embed = x_embed + self.offset

        pos_x = x_embed[:, :, :, None] / self.dim_tx
        pos_y = y_embed[:, :, :, None] / self.dim_ty
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos