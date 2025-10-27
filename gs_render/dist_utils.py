import os
import torch.distributed as dist
import torch

def is_dist():
    return "RANK" in os.environ and "LOCAL_RANK" in os.environ

def ddp_setup():
    # initialize the process group
    dist.init_process_group("nccl", init_method="env://")
    print(get_local_rank())
    torch.cuda.set_device(get_local_rank())

def ddp_cleanup():
    dist.destroy_process_group()

def get_rank():
    return dist.get_rank()

def get_local_rank():
    return int(os.environ["LOCAL_RANK"])

def get_world_size():
    return dist.get_world_size()

def get_local_world_size():
    return int(os.environ["LOCAL_WORLD_SIZE"])
