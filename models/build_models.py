from models.backbones import *
from models.heads import *
import torch.nn as nn
import torch
import os
from models.base_model import BaseSegModel


head_dict = {
    'FPNHead': FPNHead,
    'MaskRCNNSegmentationHead': MaskRCNNHeads,
    'SegFormerHead': SegFormerHead,
    'UPerHead': UPerHead
}


class SegmentationModel(BaseSegModel):
    def __init__(self, backbone: str = 'MiT-B0', pretrained_backbone='', num_classes: int = 19,
                 seg_head: str = 'UPerHead', aux_for_deeplab: bool = False, **kwargs):
        super(SegmentationModel, self).__init__(backbone=backbone, num_classes=num_classes, seg_head=seg_head, **kwargs)
        self.backbone_name = backbone
        self.head_name = seg_head
        self.aux_for_deeplab = aux_for_deeplab

        if 'MiT' in self.backbone_name:
            backbone, variant = backbone.split('-')
            self.backbone = eval(backbone)(variant)
        else:
            self.backbone = eval(self.backbone_name + '()')

        if 'mobilenetv4' in self.backbone_name:
            self.spec = MODEL_SPECS[self.backbone_name]

            first_channel = self.spec["conv0"]["block_specs"][0][1]
            second_channel = self.spec["layer1"]["block_specs"][-1][1]
            third_channel = self.spec["layer2"]["block_specs"][-1][1]
            forth_channel = self.spec["layer3"]["block_specs"][-1][1]
            fifth_channel = self.spec["layer5"]["block_specs"][0][1]
            self.backbone.channels = [first_channel, second_channel, third_channel, forth_channel, fifth_channel]
            # print(self.backbone.channels)


        if 'MiT' in backbone:
            self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in variant or 'B1' in variant else 768,
                                        num_classes)

        if 'deeplabv3' in seg_head.lower():
            self.decode_head = DeepLabV3(self.backbone.channels[-1],
                                         self.backbone.channels[-2],
                                         num_classes,
                                         self.aux_for_deeplab)
        else:
            self.decode_head = head_dict[seg_head](self.backbone.channels, 128 if 'tiny' in backbone or 'small' in backbone else 768,
                                     num_classes)

        if pretrained_backbone:
            if os.path.exists(pretrained_backbone):
                self.backbone.load_state_dict(torch.load(pretrained_backbone, map_location='cpu'), strict=False)
            else:
                print('The pretrained weights path of backbone is wrong! File does not exists!!')

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


# if __name__ == '__main__':
    # model = SegmentationModel('convnextv2_tiny', seg_head='deeplabv3').cuda()
    # model = SegmentationModel('caformer_m36').cuda()
#     # ckpt = torch.load('../segformer.b2.1024x1024.city.160k.pth')['state_dict']
#     # # TODO: Must remember that delete the additional layer(decode_head.conv_seg) when you load the segformer-mit models weight
#     # del ckpt['decode_head.conv_seg.weight']
#     # del ckpt['decode_head.conv_seg.bias']
#
#     # print(ckpt.keys())
#     # model.load_state_dict(ckpt)
#     x = torch.randn(2, 3, 384, 384).cuda()
#     model.eval()
#     y = model(x)
    # print(model)
    # print(y.shape)
#     # print(model.modules)
#     print('pass!')