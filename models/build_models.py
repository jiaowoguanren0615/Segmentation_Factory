from models.backbones import *
from models.heads import *
import torch.nn as nn
import torch

class SegmentationModel(nn.Module):  ##crossformer_small
    def __init__(self, backbone: str = 'MiT-B0', pretrained_backbone=None, num_classes: int = 19):
        super(SegmentationModel, self).__init__()
        self.backbone_name = backbone
        if 'MiT' in self.backbone_name:
            backbone, variant = backbone.split('-')
            self.backbone = eval(backbone)(variant)
        else:
            self.backbone = eval(self.backbone_name + '()')

        if pretrained_backbone is not None:
            self.backbone.load_state_dict(torch.load(pretrained_backbone, map_location='cpu'), strict=False)

        if 'MiT' in backbone:
            self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768,
                                        num_classes)
        else:
            self.decode_head = UPerHead(self.backbone.channels, 128 if 'tiny' in backbone or 'small' in backbone else 768,
                                     num_classes)


    def __str__(self):
        if 'MiT' in self.backbone_name:
            return f'SegFormer-{self.backbone_name}'
        else:
            return f'{self.backbone_name}_UPerNet'

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


if __name__ == '__main__':
    model = SegmentationModel('MiT-B0')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(model)
    print(y.shape)
    print(model.modules)
