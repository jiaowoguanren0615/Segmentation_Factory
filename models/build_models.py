from models.backbones import *
from models.heads import *
import torch.nn as nn
import torch
import os


head_dict = {
    'FPNHead': FPNHead,
    'MaskRCNNSegmentationHead': MaskRCNNHeads,
    'SegFormerHead': SegFormerHead,
    'UPerHead': UPerHead,
}


class SegmentationModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', pretrained_backbone='', num_classes: int = 19,
                 seg_head: str = 'UPerHead', **kwargs):
        super(SegmentationModel, self).__init__()
        self.backbone_name = backbone
        self.head_name = seg_head

        if 'MiT' in self.backbone_name:
            backbone, variant = backbone.split('-')
            self.backbone = eval(backbone)(variant)
        else:
            self.backbone = eval(self.backbone_name + '()')

        # print(self.backbone.channels) # [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192]

        if 'MiT' in backbone:
            self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in variant or 'B1' in variant else 768,
                                        num_classes)
        else:
            self.decode_head = head_dict[seg_head](self.backbone.channels, 128 if 'tiny' in backbone or 'small' in backbone else 768,
                                     num_classes)


        if pretrained_backbone:
            if os.path.exists(pretrained_backbone):
                self.backbone.load_state_dict(torch.load(pretrained_backbone, map_location='cpu'), strict=False)
            else:
                print('The pretrained weights path of backbone is wrong! File does not exists!!')


    def __str__(self):
        if 'MiT' in self.backbone_name:
            return f'SegFormer-{self.backbone_name}'
        else:
            return f'{self.backbone_name}_{self.head_name}'

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


if __name__ == '__main__':
    # model = SegmentationModel('kat_tiny_swish_patch16_224', seg_head='UPerHead').cuda()
    # model = SegmentationModel('MiT-B2').cuda()
    # ckpt = torch.load('../segformer.b2.1024x1024.city.160k.pth')['state_dict']
    # # TODO: Must remember that delete the additional layer(decode_head.conv_seg) when you load the segformer-mit models weight
    # del ckpt['decode_head.conv_seg.weight']
    # del ckpt['decode_head.conv_seg.bias']

    # print(ckpt.keys())
    # model.load_state_dict(ckpt)
    # x = torch.randn(2, 3, 1024, 1024).cuda()
    # model.eval()
    # y = model(x)
    # print(model)
    # print(y.shape)
    # print(model.modules)
    print('pass!')