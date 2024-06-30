""" ONNX-runtime validation script

This script was created to verify accuracy and performance of exported ONNX
models running with the onnxruntime. It utilizes the PyTorch dataloader/processing
pipeline for a fair comparison against the originals.

Copyright 2020 Ross Wightman
"""
import argparse
import numpy as np
import torch
import onnxruntime
from util.utils import AverageMeter
import time
from datasets import build_dataset
from util.metrics import Metrics
import util.utils as utils



parser = argparse.ArgumentParser(description='Pytorch ONNX Validation')
parser.add_argument("--data_root", type=str, default='/mnt/d/CityScapesDataset',
                        help="path to CityScapesDataset Dataset")
parser.add_argument('--nb-classes', type=int, default=19,
                    help='Number classes in datasets')
parser.add_argument('--onnx-input', default='./SegFormer-MiT-B0_optim.onnx', type=str, metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-output-opt', default='', type=str, metavar='PATH',
                    help='path to output optimized onnx graph')
parser.add_argument('--profile', action='store_true', default=False,
                    help='Enable profiler output.')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4), as same as the train_batch_size in train_gpu.py')
parser.add_argument('--image_size', default=518, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of datasets')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of datasets')
parser.add_argument('--crop-pct', type=float, default=None, metavar='PCT',
                    help='Override default crop pct of 0.875')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--tf-preprocessing', dest='tf_preprocessing', action='store_true',
                    help='use tensorflow mnasnet preporcessing')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    args.input_size = args.img_size

    # Set graph optimization level
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.profile:
        sess_options.enable_profiling = True
    if args.onnx_output_opt:
        sess_options.optimized_model_filepath = args.onnx_output_opt

    session = onnxruntime.InferenceSession(args.onnx_input, sess_options)


    _, val_set = build_dataset(args)

    loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False
    )

    input_name = session.get_inputs()[0].name

    batch_time = AverageMeter()
    end = time.time()

    metric = Metrics(args.nb_classes, ignore_label=255, device='cpu')
    confmat = utils.ConfusionMatrix(args.nb_classes)

    for i, (input, target) in enumerate(loader):
        # run the net and return prediction
        output = session.run([], {input_name: input.data.numpy()})
        output = output[0]  ## shape: [Batch_size, nb_classes, img_size, img_size]

        confmat.update(target.flatten(), output.argmax(1).flatten())
        metric.update(output, target.flatten())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {(input.size(0) / batch_time.avg):.3f}/s, {(100 * batch_time.avg / input.size(0)):.3f} ms/sample) \t'
                  f'val_meanF1: {metric.compute_f1()[1]}\t'
                  f'val_meanACC: {metric.compute_pixel_acc()[1]}\t'
                  f'val_mIOU: {round((confmat.compute()[2].mean().item() * 100), 2)}\t'
                )

    mean_iou = confmat.compute()[2].mean().item() * 100
    mean_iou = round(mean_iou, 2)
    all_f1, mean_f1 = metric.compute_f1()
    all_acc, mean_acc = metric.compute_pixel_acc()
    print(f"**val_meanF1: {mean_f1}\n**val_meanACC: {mean_acc}\n**val_mIOU: {mean_iou}")

if __name__ == '__main__':
    main()