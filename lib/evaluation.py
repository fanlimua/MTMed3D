import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance, compute_surface_dice

import torch
import torch.nn as nn

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3) * SR.size(4)
    acc = float(corr) / float(tensor_size)

    return acc

def get_accuracy_val(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc

def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = torch.sum((SR == 1) & (GT == 1)).float()
    FN = torch.sum((SR == 0) & (GT == 1)).float()
    SE = TP / (TP + FN + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TN = torch.sum((SR == 0) & (GT == 0)).float()
    FP = torch.sum((SR == 1) & (GT == 0)).float()
    # TN : True Negative
    # FP : False Positive

    SP = TN / (TN + FP + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = torch.sum((SR == 1) & (GT == 1)).float()
    FP = torch.sum((SR == 1) & (GT == 0)).float()
    PC = TP / (TP + FP + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    Inter = torch.sum((SR == 1) & (GT == 1)).float()
    Union = torch.sum((SR == 1) | (GT == 1)).float()

    JS = Inter / (Union + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR == 1) & (GT == 1)).float()
    DC = 2 * Inter / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC

def get_HD(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    dist = compute_hausdorff_distance(SR, GT)
    return dist

def get_NSD(SR, GT,threshold=0.5):
    # NSD : Normalized Surface Dice 
    SR = SR[0]
    GT = GT[0]
    threshold = torch.tensor(threshold)
    SR = (SR >= threshold).float()
    GT = (GT >= threshold).float()
    SR = nn.functional.one_hot(SR)
    GT = nn.functional.one_hot(GT)
    nsd = compute_surface_dice(SR, GT, threshold)
    return nsd

def sensitivity_specificity(output, target):
    with torch.no_grad():
        TP = ((output == 1) & (target == 1)).sum().item()
        TN = ((output == 0) & (target == 0)).sum().item()
        FP = ((output == 1) & (target == 0)).sum().item()
        FN = ((output == 0) & (target == 1)).sum().item()

        acc = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        return acc, sensitivity, specificity

