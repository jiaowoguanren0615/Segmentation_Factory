# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import os

import torch
import torch.distributed

from models.layers import list_mean, list_sum

__all__ = [
    "dist_init",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_barrier",
    "get_dist_local_rank",
    "sync_tensor",
    "AverageMeter",
]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: torch.Tensor or int or float) -> torch.Tensor or int or float:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: torch.Tensor or int or float, delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> torch.Tensor or int or float:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg


def dist_init() -> None:
    try:
        torch.distributed.init_process_group(backend="nccl")
        assert torch.distributed.is_initialized()
    except Exception:
        # use torchpack
        from torchpack import distributed as dist

        dist.init()
        os.environ["RANK"] = f"{dist.rank()}"
        os.environ["WORLD_SIZE"] = f"{dist.size()}"
        os.environ["LOCAL_RANK"] = f"{dist.local_rank()}"


def get_dist_rank() -> int:
    return int(os.environ["RANK"])


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    torch.distributed.barrier()


def get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def sync_tensor(tensor: torch.Tensor or float, reduce="mean") -> torch.Tensor or list[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list