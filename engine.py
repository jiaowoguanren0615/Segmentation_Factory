import torch
import torch.nn as nn
import math
import sys
from util.losses import build_target, dice_loss
from util.metrics import Metrics
from util import utils


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
    return loss


def train_one_epoch(model, optimizer, dataloader,
                    epoch, device, print_freq, clip_grad, clip_mode, loss_scaler, writer=None, args=None):
    model.train()

    num_steps = len(dataloader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if args.nb_classes == 2:
        # TODO set CrossEntropy loss-weights for object & background according to your datasets
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for idx, (img, lbl) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        img = img.to(device)
        lbl = lbl.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(img)
            loss = criterion(logits, lbl, loss_weight, num_classes=args.nb_classes, dice=args.dice, ignore_index=args.ignore_index)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)


        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss_value, lr=lr)

        if idx % print_freq == 0:
            if args.local_rank == 0:
                iter_all_count = epoch * num_steps + idx
                writer.add_scalar('train_loss', loss, iter_all_count)
                writer.add_scalar('train_lr', lr, iter_all_count)

    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()
    return metric_logger.meters["loss"].global_avg, lr



@torch.inference_mode()
def evaluate(args, model, dataloader, device, print_freq, writer=None):
    model.eval()

    metric = Metrics(args.nb_classes, args.ignore_label, args.device)
    confmat = utils.ConfusionMatrix(args.nb_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for idx, (images, labels) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast(): # TODO: ConfusionMatrix not implemented for 'Half' data
        outputs = model(images)
        confmat.update(labels.flatten(), outputs.argmax(1).flatten())
        metric.update(outputs, labels.flatten())

        if writer:
            if idx % print_freq == 0 & args.local_rank == 0:
                writer.add_scalar('valid_mf1', metric.compute_f1()[1])
                writer.add_scalar('valid_acc', metric.compute_pixel_acc()[1])
                writer.add_scalar('valid_mIOU', metric.compute_iou()[1])


    confmat.reduce_from_all_processes()
    metric.reduce_from_all_processes()

    torch.cuda.empty_cache()

    return confmat, metric