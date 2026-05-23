import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from lpips import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths



# Data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])



def epe_loss(output, target):
    if output.numel() == 0 or target.numel() == 0 or output.shape != target.shape:  return 0
    return torch.mean(torch.norm(output - target, p=2, dim=1))

def abs_real_loss(output, target):
    if output.numel() == 0 or target.numel() == 0 or output.shape != target.shape:  return 0
    return torch.mean( torch.abs(output - target) / (target+1e-10) )
   

def calculate_psnr(output, target):
    output = output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
    target = target.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
    psnrs = [psnr_metric(target[i], output[i]) for i in range(output.shape[0])]
    return sum(psnrs) / len(psnrs)

def calculate_ssim(output, target):
    output = output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
    target = target.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
    ssims = [
        ssim_metric(target[i].mean(axis=2), output[i].mean(axis=2), data_range=1.0)
        for i in range(output.shape[0])
    ]
    return sum(ssims) / len(ssims)


 

 

