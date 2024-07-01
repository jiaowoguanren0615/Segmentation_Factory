from .voc import VOCSegmentation
from .ade import ADE20K
from .cityscapes import Cityscapes
from .coco_stuff import COCOStuff
from datasets import extra_transform as et
from torchvision import transforms as T




def build_dataset(args):
    train_transform = et.ExtCompose([
        et.ExtRandomCrop(size=(args.image_size, args.image_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        # et.ExtRandomCrop(size=(args.image_size, args.image_size)),
        et.ExtResize(args.image_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    assert args.dataset.lower() in ['cityscapes', 'voc', 'cocostuff', 'ade'], 'No support training dataset!'

    if args.dataset.lower() == 'voc':
        args.data_root = '/mnt/d/'
        train_dst = VOCSegmentation(root=args.data_root, split='train',
                                    transform=train_transform)
        val_dst = VOCSegmentation(root=args.data_root, split='val',
                                  transform=val_transform)
    elif args.dataset.lower() == 'cityscapes':
        train_dst = Cityscapes(root=args.data_root, split='train',
                               transform=train_transform)
        val_dst = Cityscapes(root=args.data_root, split='val',
                             transform=val_transform)
    elif args.dataset.lower() == 'cocostuff':
        args.data_root = '/mnt/d/CocoStuff2017'
        train_dst = COCOStuff(root=args.data_root, split='train',
                               transform=train_transform)
        val_dst = COCOStuff(root=args.data_root, split='val',
                             transform=val_transform)
    elif args.dataset.lower() == 'ade':
        args.data_root = '/mnt/d/ADEChallengeData2016'
        train_dst = ADE20K(root=args.data_root, split='train',
                              transform=train_transform)
        val_dst = ADE20K(root=args.data_root, split='val',
                            transform=val_transform)

    return train_dst, val_dst