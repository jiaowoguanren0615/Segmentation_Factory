import os
import sys
import tarfile
import collections

import torch
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import io

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The datasets year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the datasets from the internet and
            puts it in root directory. If datasets is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()

    PALETTE = torch.tensor([
        (0, 0, 0),  # Background
        (128, 0, 0),  # Aeroplane
        (0, 128, 0),  # Bicycle
        (128, 128, 0),  # Bird
        (0, 0, 128),  # Boat
        (128, 0, 128),  # Bottle
        (0, 128, 128),  # Bus
        (128, 128, 128),  # Car
        (64, 0, 0),  # Cat
        (192, 0, 0),  # Chair
        (64, 128, 0),  # Cow
        (192, 128, 0),  # Dining Table
        (64, 0, 128),  # Dog
        (192, 0, 128),  # Horse
        (64, 128, 128),  # Motorbike
        (192, 128, 128),  # Person
        (0, 64, 0),  # Potted Plant
        (128, 64, 0),  # Sheep
        (0, 192, 0),  # Sofa
        (128, 192, 0),  # Train
        (0, 64, 128)  # TV/Monitor
    ])

    ## VOC2010 PALETTE
    # PALETTE = torch.tensor([
    #     [180, 120, 120], [6, 230, 230], [80, 50, 50],
    #     [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    #     [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    #     [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    #     [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    #     [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    #     [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    #     [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    #     [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    #     [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    #     [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    #     [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    #     [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    #     [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    #     [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]
    # ])

    def __init__(self,
                 root,
                 year='2012',
                 split='train',
                 download=False,
                 transform=None):

        is_aug = False
        if year == '2012_aug':
            is_aug = True
            year = '2012'

        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform

        self.image_set = split
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if is_aug and self.image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(
                mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join(self.root, 'train_aug.txt')  # './datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        ## TODO If you want to use 'visualize_dataset_sample' funtion, comment follow two lines and do step 2
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])


        ## TODO step 2: Comment these follow four lines code
        # img_path = str(self.images[index])
        # lbl_path = str(self.masks[index])
        # image = io.read_image(img_path)
        # target = io.read_image(lbl_path)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target.long()

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        """
        for i in range(len(images)):
            image = images[i].detach().cpu().numpy()
            target = targets[i]
            pred = preds[i]
            print(target.shape)

            image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
            target = loader.dataset.decode_target(target).astype(np.uint8)
            pred = loader.dataset.decode_target(pred).astype(np.uint8)

            Image.fromarray(image).save('results/%d_image.png' % img_id)
            Image.fromarray(target).save('results/%d_target.png' % img_id)
            Image.fromarray(pred).save('results/%d_pred.png' % img_id)
        """
        return cls.cmap[mask]


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)


# if __name__ == '__main__':
#     from datasets.visualize import visualize_dataset_sample
#     visualize_dataset_sample(VOCSegmentation, '/mnt/d/')
