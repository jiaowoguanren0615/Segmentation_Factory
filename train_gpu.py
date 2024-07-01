import os
import re
import torch
import numpy as np
import datetime
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from timm.utils import NativeScaler
from timm.models import create_model
from timm.optim import create_optimizer

import torch.backends.cudnn as cudnn

from models.backbones import *
from models.heads import *
from models.build_models import SegmentationModel

from datasets import *

from util import utils

from scheduler import create_scheduler

from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Segmentation Models training and evaluation script', add_help=False)

    # Dataset parameters
    parser.add_argument("--data_root", type=str, default='/mnt/d/CityScapesDataset',
                        help="path to CityScapes Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes', 'voc', 'cocostuff', 'ade'])
    parser.add_argument("--image_size", type=int, default=512, help="input size")
    parser.add_argument("--ignore_label", type=int, default=255, help="the dataset ignore_label")
    parser.add_argument("--ignore_index", type=int, default=255, help="the dataset ignore_index")
    parser.add_argument("--dice", type=bool, default=True, help="Calculate Dice Loss")
    parser.add_argument('--data_len', default=5000, type=int,
                        help='count of your entire data_set. For example: Cityscapes 5000, voc 11530')
    parser.add_argument('--nb_classes', default=19, type=int,
                        choices=[19, 21, 172, 151],
                        help='number classes of your dataset (including background)'
                             'CityScapes: 19'
                             'VOC2012: 21'
                             'cocostuff: 172'
                             'ADE20K: 151'
                        )

    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validation (default: 1)')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument("--train_print_freq", type=int, default=100)
    parser.add_argument("--val_print_freq", type=int, default=100)

    # Model parameters
    parser.add_argument('--backbone', default='MiT-B0', type=str, metavar='MODEL',
                        choices=['crossformer_tiny', 'crossformer_base', 'crossformer_large', 'crossformer_small',
                                 'crossformerpp_base', 'crossformerpp_large', 'crossformerpp_small', 'crossformerpp_huge',
                                 'MiT-B0', 'MiT-B1', 'MiT-B2', 'MiT-B3', 'MiT-B4', 'MiT-B5',
                                 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge', 'convnextv2_tiny'],
                        help='Feature extractor Backbones')

    parser.add_argument('--pretrained_backbone', default='', type=str, metavar='MODEL',
                        help='Backbone pretrained weights path')

    parser.add_argument('--heads', default='segformer', type=str, metavar='MODEL',
                        choices=['fpn', 'segformer', 'upernet'],
                        help='Segmentation Head')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')


    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-ep', action='store_true', default=False,
                        help='using the epoch-based scheduler')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=2e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-4, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                        help='list of decay epoch indices for multistep lr. must be increasing')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')


    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--encoder_pretrain_weights', type=str, default='/mnt/d/PythonCode/DPT/DepthAnythingV2/dinov2_vits14_pretrain.pth')
    parser.add_argument('--freeze_layers', type=bool, default=True, help='freeze layers')
    parser.add_argument('--set_bn_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')


    parser.add_argument('--save_weights_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--writer_output', default='./',
                        help='path where to save SummaryWriter, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')
    return parser


def main(args):
    print(args)
    utils.init_distributed_mode(args)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.writer_output, 'runs'))

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # start = time.time()
    best_mIoU = 0.0
    best_F1 = 0.0
    best_acc = 0.0
    device = args.device

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_set, valid_set = build_dataset(args)

    if args.distributed:
        sampler_train = DistributedSampler(train_set, num_replicas=utils.get_world_size(), rank=utils.get_rank(),
                                           shuffle=True)
        sampler_val = DistributedSampler(valid_set)
    else:
        sampler_train = RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(valid_set)

    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=True, pin_memory=args.pin_mem, sampler=sampler_train)

    valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                           drop_last=True, pin_memory=args.pin_mem, sampler=sampler_val)


    model = SegmentationModel(args.backbone,
                              pretrained_backbone=args.pretrained_backbone,
                              num_classes=args.nb_classes,
                              args=args)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.load_model(args.finetune, model)

        checkpoint_model = checkpoint
        # state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            if 'scratch.output_conv2' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if 'scratch.output_conv2' not in name:
                    para.requires_grad_(False)
                else:
                    print('training {}'.format(name))

    model.to(device)

    n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('\n********ESTABLISH ARCHITECTURE********')
    print(f'Model: {model}\nNumber of parameters: {n_parameters}')
    print('**************************************\n')

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.save_weights_dir)
    if args.save_weights_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    if args.save_weights_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")

    checkpoint_name = utils.get_pth_file(args.save_weights_dir)

    if args.resume or checkpoint_name:
        args.resume = os.path.join(f'{args.save_weights_dir}/', checkpoint_name)
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model_state'])
        print(msg)
        if not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for state in optimizer.state.values():  # load parameters to cuda
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
            best_mIoU = checkpoint['best_mIoU']
            best_F1 = checkpoint['F1_Score']
            best_acc = checkpoint['Acc']
            print(f'Now max mIOU is {best_mIoU}\n')
            print(f'Now max F1-score is {best_F1}\n')
            print(f'Now max Accuracy is {best_acc}\n')
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        # util.replace_batchnorm(model) # Users may choose whether to merge Conv-BN layers during eval
        print(f"Evaluating model: {args.model}")
        confmat, metric = evaluate(args, model, valloader, device, args.val_print_freq)
        mean_iou = confmat.compute()[2].mean().item() * 100
        mean_iou = round(mean_iou, 2)
        all_f1, mean_f1 = metric.compute_f1()
        all_acc, mean_acc = metric.compute_pixel_acc()
        print(f"**val_meanF1: {mean_f1}\n**val_meanACC: {mean_acc}\n**val_mIOU: {mean_iou}")

    print(f"Start training for {args.epochs} epochs")

    for epoch in range(args.epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch(model, optimizer, trainloader,
                                        epoch, device, args.train_print_freq, args.clip_grad, args.clip_mode,
                                        loss_scaler, writer, args)

        confmat, metric = evaluate(args, model, valloader, device, args.val_print_freq, writer)
        mean_iou = confmat.compute()[2].mean().item() * 100
        mean_iou = round(mean_iou, 2)
        all_f1, mean_f1 = metric.compute_f1()
        all_acc, mean_acc = metric.compute_pixel_acc()
        print(f"**Val_meanF1: {mean_f1}\n**Val_meanACC: {mean_acc}\n**Val_mIOU: {mean_iou}")

        lr_scheduler.step(epoch)

        val_info = f'{str(confmat)}\nval_meanF1: {mean_f1}\nval_meanACC: {mean_acc}'

        print(val_info)

        if utils.is_main_process():
            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

        if mean_iou > best_mIoU:
            print(f'Increasing mIoU: from {best_mIoU} to {mean_iou}!\n')
            best_mIoU = mean_iou
            print(f'Max mIOU: {best_mIoU}\n')
            if utils.is_main_process():
                checkpoint_save = {
                    "model_state": model_without_ddp.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": lr_scheduler.state_dict(),
                    "best_mIoU": mean_iou,
                    "F1_Score": mean_f1,
                    "Acc": mean_acc,
                    "scaler": loss_scaler.state_dict()
                }
                torch.save(checkpoint_save, f'{args.save_weights_dir}/{model}_best_model.pth')
                print('******************Save Checkpoint******************')
                print(f'Save weights to {args.save_weights_dir}/{model}_best_model.pth\n')
        else:
            print('*********No improving mIOU, No saving checkpoint*********')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Segmentation Models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_weights_dir:
        Path(args.save_weights_dir).mkdir(parents=True, exist_ok=True)
    main(args)