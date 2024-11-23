import torch
import argparse
import os
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from models.build_models import SegmentationModel
from datasets import Cityscapes, VOCSegmentation, COCOStuff, ADE20K
from datasets.visualize import draw_text
from rich.console import Console
from PIL import Image


console = Console()


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Segmentation Models Evaluation Script', add_help=False)

    # Dataset parameters
    parser.add_argument("--data_root", type=str, default='/mnt/d/CityScapesDataset/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
                        help="path to CityScapes Dataset for prediction")
    parser.add_argument("--dataset", type=str, default='Cityscapes',
                        choices=['Cityscapes', 'VOCSegmentation', 'COCOStuff', 'ADE20K'])
    parser.add_argument("--img_size", type=int, default=1024, help="input size")

    # Model parameters
    parser.add_argument('--backbone', default='MiT-B2', type=str, metavar='MODEL',
                        choices=['crossformer_tiny', 'crossformer_base', 'crossformer_large', 'crossformer_small',
                                 'crossformerpp_base', 'crossformerpp_large', 'crossformerpp_small', 'crossformerpp_huge',
                                 'MiT-B0', 'MiT-B1', 'MiT-B2', 'MiT-B3', 'MiT-B4', 'MiT-B5',
                                 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge', 'convnextv2_tiny'],
                        help='Feature extractor Backbones')

    parser.add_argument('--heads', default='SegFormerHead', type=str, metavar='MODEL',
                        choices=['FPNHead', 'MaskRCNNSegmentationHead', 'SegFormerHead', 'UPerHead'],
                        help='Segmentation Head')

    parser.add_argument('--weight_file', default='./segformer.b2.1024x1024.city.160k.pth',
                        help='model weight file')
    parser.add_argument('--save_pred_picture', default='./predict',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    return parser

class SemSeg:
    def __init__(self, args=None) -> None:
        # inference device cuda or cpu
        self.device = args.device

        # get dataset classes' colors and labels
        self.palette = eval(args.dataset).PALETTE
        self.labels = eval(args.dataset).CLASSES

        # initialize the model and load weights and send to device
        self.model = SegmentationModel(args.backbone)
        ckpt = torch.load(args.weight_file)['state_dict']
        del ckpt['decode_head.conv_seg.weight']
        del ckpt['decode_head.conv_seg.bias']
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = [args.img_size, args.img_size]
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Image:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = draw_text(seg_image, seg_map, self.labels)
        return image

    @torch.inference_mode()
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def predict(self, img_fname: str, overlay: bool = True) -> Image:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map



def main(args):
    test_file = Path(args.data_root)
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    save_dir = Path(args.save_pred_picture) / f'{args.dataset}_test_results'
    save_dir.mkdir(exist_ok=True)

    semseg = SemSeg(args)

    with console.status("[bright_green]Processing..."):
        if test_file.is_file():
            console.rule(f'[green]{test_file}')
            segmap = semseg.predict(str(test_file), True)
            segmap.save(save_dir / f"{str(test_file.stem)}.png")
        else:
            files = test_file.glob('*.*')
            for file in files:
                console.rule(f'[green]{file}')
                segmap = semseg.predict(str(file), True)
                segmap.save(save_dir / f"{str(file.stem)}.png")

    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Segmentation Models Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()

    if not os.path.exists(args.save_pred_picture):
        os.mkdir(args.save_pred_picture)

    main(args)

    torch.cuda.empty_cache()