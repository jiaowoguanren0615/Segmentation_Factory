import torch.nn as nn


class BaseSegModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19,
                 seg_head: str = 'UPerHead', **kwargs):
        super(BaseSegModel, self).__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.head_name = seg_head


    def __str__(self):
        if 'MiT' in self.backbone_name:
            return f'SegFormer-{self.backbone_name}'
        else:
            return f'{self.backbone_name}_{self.head_name}'