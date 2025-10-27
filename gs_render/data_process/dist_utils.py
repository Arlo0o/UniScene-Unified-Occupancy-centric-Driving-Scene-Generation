import os
import torch.distributed as dist
import torch

def is_dist():
    return "RANK" in os.environ and "LOCAL_RANK" in os.environ

def ddp_setup():
    # initialize the process group
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device('cuda:'+get_local_rank())

def ddp_cleanup():
    dist.destroy_process_group()

def get_rank():
    return dist.get_rank()

def get_local_rank():
    return os.environ["LOCAL_RANK"]
    #return dist.get_local_rank()

def get_world_size():
    return dist.get_world_size()