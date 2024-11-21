import torch
from PIL import Image
import glob
import os
import numpy as np
from torch.utils.data import Dataset

from random import sample
import datasets.extra_transform as T



class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, args, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = args.img_size
    crop_size = args.img_size

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(args.img_size, mean=mean, std=std)


# def preprocess(x):
#     return x / 255
#
#
# def separate_class(x):
#     first_dim = torch.where(x == 0, torch.ones_like(x), torch.zeros_like(x))
#     second_dim = torch.where(x == 1, torch.ones_like(x), torch.zeros_like(x))
#     #     third_dim = torch.where(x == 2, torch.ones_like(x), torch.zeros_like(x))
#     return torch.cat((first_dim, second_dim)) / 1
#
#
#
# transform = transforms.Compose([
#     transforms.PILToTensor(),
#     transforms.Resize((224, 224)),
#     preprocess,
# ])
#
# target_transform = transforms.Compose([
#     # transforms.PILToTensor(),
#     transforms.Resize((224, 224)),
#     separate_class
# ])
#
# trans = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     transforms.RandomRotation(30),
#     transforms.CenterCrop(224)
# ])




class KvasirDataSet(Dataset):
    def __init__(self, Kvasir_folder, ClinicDB_folder, img_size=224, train_mode=False, transform=None):

        super(KvasirDataSet, self).__init__()
        self.img_size = img_size
        self.train_mode = train_mode
        self.transform = transform

        self.img_files1 = glob.glob(os.path.join(Kvasir_folder, 'images', '*.jpg'))
        self.img_files2 = glob.glob(os.path.join(ClinicDB_folder, 'Original', '*.png'))

        if self.train_mode:
            self.img_files1 = self.img_files1
            self.img_files2 = self.img_files2

        else:  # use random 20% dataset for valid_data
            self.img_files1 = sample(self.img_files1, len(self.img_files1) // 5)
            self.img_files2 = sample(self.img_files2, len(self.img_files2) // 5)

        self.mask_files1 = []
        for img_path in self.img_files1:
            self.mask_files1.append(os.path.join(Kvasir_folder, 'masks', os.path.basename(img_path)))
        self.mask_files2 = []
        for img_path in self.img_files2:
            self.mask_files2.append(os.path.join(ClinicDB_folder, 'Ground Truth', os.path.basename(img_path)))

    def __getitem__(self, index):

        if index < len(self.img_files1):
            img_path = self.img_files1[index]
            mask_path = self.mask_files1[index]
            data = Image.open(img_path).convert('RGB')
            label = Image.open(mask_path).convert('L')
            label = np.array(label) / 255
            mask = Image.fromarray(label)

            if self.transform is not None:
                data, mask = self.transform(data, mask)
            return data, mask

        else:
            index = index - len(self.img_files1)
            img_path = self.img_files2[index]
            mask_path = self.mask_files2[index]
            data = Image.open(img_path).convert('RGB')
            label = Image.open(mask_path).convert('L')
            label = np.array(label) / 255
            mask = Image.fromarray(label)

            if self.transform is not None:
                data, mask = self.transform(data, mask)
            return data, mask

    def __len__(self):
        return len(self.img_files1) + len(self.img_files2)


# if __name__ == '__main__':
#     train_ds = KvasirDataSet(
#         "/mnt/d/MedicalSeg/Kvasir-SEG/",
#         "/mnt/d/MedicalSeg/CVC-ClinicDB/",
#         train_mode=True
#     )
#     train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
#     print(len(train_ds))  # 1612
#     print(len(train_loader)) # 51
#
#     valid_ds = KvasirDataSet(
#         "/mnt/d/MedicalSeg/Kvasir-SEG/",
#         "/mnt/d/MedicalSeg/CVC-ClinicDB/",
#         train_mode=False
#     )
#     valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=32, shuffle=False)
#     print(len(valid_ds))  # 322
#     print(len(valid_loader))  # 11


# def build_dataset(args):
#     train_ds = KvasirDataSet(
#         args.Kvasir_path,
#         args.ClinicDB_path,
#         args.img_size,
#         train_mode=True,
#         transform=get_transform(train=True, args=args)
#     )
#
#     valid_ds = KvasirDataSet(
#         args.Kvasir_path,
#         args.ClinicDB_path,
#         args.img_size,
#         train_mode=False,
#         transform=get_transform(train=False, args=args)
#     )
#     return train_ds, valid_ds