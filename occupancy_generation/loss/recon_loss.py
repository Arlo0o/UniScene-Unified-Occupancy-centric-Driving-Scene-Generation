from __future__ import print_function, division

from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F

@OPENOCC_LOSS.register_module()
class ReconLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=-100, use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.recon_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def recon_loss(self, logits, labels):
        weight = None
        if self.use_weight:
            if self.cls_weight is not None:
                weight = self.cls_weight
            else:
                one_hot_labels = F.one_hot(labels, num_classes=logits.shape[-1]) # bs, F, H, W, D, C
                cls_freq = torch.sum(one_hot_labels, dim=[0, 1, 2, 3, 4]) # C
                weight = 1.0 / cls_freq.clamp_min_(1) * torch.numel(labels) / logits.shape[-1]
        
        rec_loss = F.cross_entropy(logits.permute(0, 5, 1, 2, 3, 4), labels, ignore_index=self.ignore, weight=weight)
        return rec_loss
    
@OPENOCC_LOSS.register_module()
class LovaszLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
    
    def lovasz_loss(self, logits, labels):
        logits = logits.flatten(0, 1).permute(0, 4, 1, 2, 3).softmax(dim=1)
        labels = labels.flatten(0, 1)
        loss = lovasz_softmax(logits, labels)
        return loss


"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return 0.#probas * 0.
    #print(probas.size())
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        #3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid]#.nonzero().squeeze()]
    # print(labels)
    # print(valid)
    vlabels = labels[valid]
    return vprobas, vlabels

# --------------------------- HELPER FUNCTIONS ---------------------------

def isnan(x):
    return x != x    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
