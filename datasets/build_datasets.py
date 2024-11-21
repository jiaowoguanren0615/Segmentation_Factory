from .voc import VOCSegmentation
from .ade import ADE20K
from .cityscapes import Cityscapes
from .coco_stuff import COCOStuff
from .kvasir import KvasirDataSet
from .synapse import Synapse_dataset
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

    # Add check the nb_classes setting, match the dataset, improve robust
    assert args.dataset.lower() in ['cityscapes', 'voc', 'cocostuff', 'ade', 'synapse', 'kvasir'], 'No support training dataset!'
    assert args.nb_classes in [19, 21, 172, 151, 9, 2], 'No support nb_classes for training dataset!'

    if args.dataset.lower() == 'voc':
        assert args.nb_classes == 21, 'Your nb_classes setting does not match PascalVOC 2012!'

        train_dst = VOCSegmentation(root=args.data_root, split='train',
                                    transform=train_transform)
        val_dst = VOCSegmentation(root=args.data_root, split='val',
                                  transform=val_transform)

    elif args.dataset.lower() == 'cityscapes':
        assert args.nb_classes == 19, 'Your nb_classes setting does not match Cityscapes!'

        train_dst = Cityscapes(root=args.data_root, split='train',
                               transform=train_transform)
        val_dst = Cityscapes(root=args.data_root, split='val',
                             transform=val_transform)

    elif args.dataset.lower() == 'cocostuff':
        assert args.nb_classes == 172, 'Your nb_classes setting does not match Cocostuff!'

        train_dst = COCOStuff(root=args.data_root, split='train',
                               transform=train_transform)
        val_dst = COCOStuff(root=args.data_root, split='val',
                             transform=val_transform)

    elif args.dataset.lower() == 'ade':
        assert args.nb_classes == 151, 'Your nb_classes setting does not match ADE20K!'

        train_dst = ADE20K(root=args.data_root, split='train',
                              transform=train_transform)
        val_dst = ADE20K(root=args.data_root, split='val',
                            transform=val_transform)

    elif args.dataset.lower() == 'kvasir':
        assert args.nb_classes == 2, 'Your nb_classes setting does not match Kvasir_ClinicDB!'

        train_dst = KvasirDataSet(
                args.Kvasir_path,
                args.ClinicDB_path,
                args.image_size,
                train_mode=True,
                transform=train_transform
            )
        val_dst = KvasirDataSet(
            args.Kvasir_path,
            args.ClinicDB_path,
            args.image_size,
            train_mode=False,
            transform=val_transform
        )

    elif args.dataset.lower() == 'synapse':
        assert args.nb_classes == 9, 'Your nb_classes setting does not match Synapse!'

        train_dst = Synapse_dataset(
            args.synapse_train_base_dir,
            args.synapse_list_dir,
            split='train',
            transform=train_transform
        )
        val_dst = Synapse_dataset(
            args.synapse_val_base_dir,
            args.synapse_list_dir,
            split='test_vol',
            transform=val_transform
        )
    return train_dst, val_dst