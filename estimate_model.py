import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from models.build_models import SegmentationModel
from datasets import Cityscapes
from datasets.visualize import draw_text
from rich.console import Console
from PIL import Image


console = Console()

class SemSeg:
    def __init__(self, cfg=None) -> None:
        # inference device cuda or cpu
        self.device = torch.device('cuda')

        # get dataset classes' colors and labels
        self.palette = eval('Cityscapes').PALETTE
        self.labels = eval('Cityscapes').CLASSES

        # initialize the model and load weights and send to device
        self.model = SegmentationModel('MiT-B2')
        ckpt = torch.load('./segformer.b2.1024x1024.city.160k.pth')['state_dict']
        del ckpt['decode_head.conv_seg.weight']
        del ckpt['decode_head.conv_seg.bias']
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = [1024, 1024]
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

    def predict(self, img_fname: str, overlay: bool) -> Image:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    # args = parser.parse_args()


    test_file = Path('./assests/Cityscapes')
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    save_dir = Path('predict') / 'test_results'
    save_dir.mkdir(exist_ok=True)

    semseg = SemSeg()

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