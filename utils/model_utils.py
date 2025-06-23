import os
import argparse
import time
import numpy as np
import torch
from lib.data_loader import test_transforms, load_decathlon_datalist, val_transforms
from lib.data_loader import train_transforms, val_transforms, load_decathlon_datalist
from monai.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils
from monai.apps.detection.metrics.coco import COCOMetric
from monai.metrics import DiceMetric
import torch.nn.functional as F
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tabulate import tabulate
from sklearn.model_selection import KFold
from ptflops import get_model_complexity_info
from lib.task_manager import *
from model.network import MultiSwin
from timm.utils import accuracy
import pandas as pd
import seaborn as sns
from train import Trainer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class ModelWrapper(nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task = task  

    def forward(self, x):
        return self.model(x, task=self.task)

def load_data_indices(full_data, index):
    """Helper function to fetch data based on indices."""
    return [full_data[i] for i in index]

def ini_models(device):
    model = MultiSwin(img_size=(96,96,96), in_channels=4)
    model.to(device)
    task_manager = TaskManager()
    return model, task_manager


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

def calculate_macs_with_ptflops(model, input_size):
    model.eval()
    with torch.no_grad():
        flops, _ = get_model_complexity_info(model, (4, input_size, input_size, input_size), as_strings=False, print_per_layer_stat=False)

    macs = flops / 2  # FLOPs are 2x MACs
    return macs, flops


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, model_name, input_size, device, task, n_runs=50):
    # Wrap the model to provide default `args` for inference
    # model = YOLOInferenceWrapper(model, device)
    model.eval()
    if model_name == "Classification branch":
        dummy_input = torch.rand(1, 3, input_size, input_size, input_size).to(device)
    else:
        dummy_input = torch.rand(1, 4, input_size, input_size, input_size).to(device)
    
    # Warm-up
    for _ in range(10):
        _ = model(dummy_input, task = task)
    
    # Measure inference time
    with torch.no_grad():
        start_time = time.time()
        for _ in range(n_runs):
            _ = model(dummy_input)
        end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    return avg_time

def get_model_size_from_pytorch(model):
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1
    }
    
    total_size_in_bytes = 0  
    for param in model.parameters():
        param_dtype = param.dtype 
        
        if param_dtype not in dtype_to_bytes:
            raise ValueError(f"Unsupported data type: {param_dtype}")
        
        param_size_in_bytes = param.numel() * dtype_to_bytes[param_dtype]
        total_size_in_bytes += param_size_in_bytes  

    size_in_mb = total_size_in_bytes / 1e6
    return size_in_mb

def load_pre_trained_weights_multiswin(root_dir, fold_name, model, task):
    """
    Load pre-trained weights for a specific task or the entire MultiSwin model.
    """
    if fold_name:
        root_dir = os.path.join(root_dir, fold_name)
    files = os.listdir(root_dir)

    for file in files:
        if task is None or (task in file and file.endswith(".pkl")):
            print(f"Loading pre-trained model for: {'full model' if task is None else task}")
            file_path = os.path.join(root_dir, file)
            state_dict = torch.load(file_path, map_location="cuda")

            # Always load shared encoder
            encoder_weights = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if "encoder." in k
            }
            model.encoder.load_state_dict(encoder_weights)
            print("Encoder weights loaded.")

            if task == "segmentation":
                model.build_seg_decoder()
                seg_weights = {
                    k.replace("seg_decoder.", ""): v
                    for k, v in state_dict.items()
                    if "seg_decoder." in k
                }
                model.seg_decoder.load_state_dict(seg_weights)

            elif task == "classification":
                model.build_seg_decoder()
                model.build_cls_decoder()
                seg_weights = {
                    k.replace("seg_decoder.", ""): v
                    for k, v in state_dict.items()
                    if "seg_decoder." in k
                }
                cls_weights = {
                    k.replace("cls_decoder.", ""): v
                    for k, v in state_dict.items()
                    if "cls_decoder." in k
                }
                model.seg_decoder.load_state_dict(seg_weights)
                model.cls_decoder.load_state_dict(cls_weights)

            elif task == "detection":
                model.build_det_decoder()
                det_weights = {
                    k.replace("det_decoder.", ""): v
                    for k, v in state_dict.items()
                    if "det_decoder." in k
                }
                model.det_decoder.load_state_dict(det_weights)

            else:
                # Load the full model if task is None or task not matched
                model.build_seg_decoder()
                model.build_cls_decoder()
                model.build_det_decoder()
                model.load_state_dict(state_dict)
                print("Full model weights loaded.")

            print(f"MultiSwin weights loaded successfully.")
            break

def load_pre_trained_weights(root_dir, fold_name, model, task=None):
    """
    Load pre-trained weights into the MultiSwin model based on the specified task.
    Encoder is always loaded if possible. Task-specific decoders are loaded based on task.
    """
    if fold_name:
        root_dir = os.path.join(root_dir, fold_name)

    files = os.listdir(root_dir)
    print("Loading Pre-trained Weights...")

    for file in files:
        file_path = os.path.join(root_dir, file)

        if not (file.endswith(".pt") or file.endswith(".pth") or file.endswith(".pkl")):
            continue

        # Always try to load encoder from seg model if available
        if "seg" in file:
            weights = torch.load(file_path, map_location="cuda")

            encoder_weights = {
                k.replace("swinViT.", ""): v
                for k, v in weights.items()
                if k.startswith("swinViT")
            }
            model.encoder.load_state_dict(encoder_weights, strict=False)
            print("Shared Encoder loaded from segmentation model.")

            if task in ["segmentation", "classification", "multi"]:
                model.build_seg_decoder()
                seg_dec_weights = {
                    k: v for k, v in weights.items()
                    if k.startswith("encoder") or k.startswith("decoder") or k.startswith("out")
                }
                model.seg_decoder.load_state_dict(seg_dec_weights, strict=False)
                print("Segmentation Decoder loaded.")

        # Load classification head if needed
        if "cls" in file and task in ["classification", "multi"]:
            weights = torch.load(file_path, map_location="cuda")

            model.build_cls_decoder()
            cls_weights = {
                k: v for k, v in weights.items()
                if k.startswith("head")
            }
            model.cls_decoder.load_state_dict(cls_weights, strict=False)
            print("Classification Decoder loaded.")

        # Load detection decoder
        if "det" in file and task in ["detection", "multi"]:
            weights = torch.load(file_path, map_location="cuda")

            model.build_det_decoder()
            det_weights = {}
            for k, v in weights.items():
                if k.startswith("feature_extractor.fpn"):
                    new_k = k.replace("feature_extractor.fpn", "fpn")
                    det_weights[new_k] = v
                elif k.startswith("classification_head") or k.startswith("regression_head"):
                    det_weights[k] = v
            model.det_decoder.load_state_dict(det_weights, strict=False)
            print("Detection Decoder loaded.")

    print(f"Weights loading complete for task: {task or 'all tasks'}")

def wrap_validation_result(task, result):

    result_dict = {
        "Dice_WT": None, "Dice_TC": None, "Dice_EH": None,
        "HD_WT": None, "HD_TC": None, "HD_ET": None,
        "Accuracy": None, "Sen(pos)": None, "Spe(neg)": None,
        "mAP_IoU": None, "mAR_IoU": None
    }
    det_metric_dict = {}

    if task == "segmentation":
        val_loss, val_metric, metric_tc, metric_wt, metric_et, hd95_metric_mean, hd95_tc, hd95_wt, hd95_et = result
        result_dict.update({
            "Dice_WT": metric_wt, "Dice_TC": metric_tc, "Dice_EH": metric_et,
            "HD_WT": hd95_wt, "HD_TC": hd95_tc, "HD_ET": hd95_et
        })

    elif task == "classification":
        cls_val_loss, val_acc, sensitivity, specificity = result
        result_dict.update({
            "Accuracy": val_acc, "Sen(pos)": sensitivity, "Spe(neg)": specificity
        })

    elif task == "detection":
        val_epoch_loss, val_epoch_cls_loss, val_epoch_box_reg_loss, val_epoch_metric_dict = result
        result_dict.update({
            "mAP_IoU": val_epoch_metric_dict.get("mAP_IoU_0.10_0.50_0.05_MaxDet_10"),
            "mAR_IoU": val_epoch_metric_dict.get("mAR_IoU_0.10_0.50_0.05_MaxDet_10")
        })
        det_metric_dict = val_epoch_metric_dict

    elif task == "multi":
        val_epoch_results, det_metric_dict = result

        # Extract main metrics
        result_dict.update({
            "Dice_WT": val_epoch_results.get("metric_wt"),
            "Dice_TC": val_epoch_results.get("metric_tc"),
            "Dice_EH": val_epoch_results.get("metric_et"),
            "HD_WT": val_epoch_results.get("hd_wt"),
            "HD_TC": val_epoch_results.get("hd_tc"),
            "HD_ET": val_epoch_results.get("hd_et"),
            "Accuracy": val_epoch_results.get("cl_acc"),
            "Sen(pos)": val_epoch_results.get("sensitivity"),
            "Spe(neg)": val_epoch_results.get("specificity")
        })

        det_summary = val_epoch_results.get("val_det_epoch_metric_dict", {})
        result_dict.update({
            "mAP_IoU": det_summary.get("mAP_IoU_0.10_0.50_0.05_MaxDet_10"),
            "mAR_IoU": det_summary.get("mAR_IoU_0.10_0.50_0.05_MaxDet_10")
        })

    return result_dict, det_metric_dict

def process_multi_task_outputs(inference_data, seg_outputs, cls_outputs, det_outputs, 
                              seg_labels, cl_label, det_targets, task_manager, det_metric,
                              inference_img_filenames, result_path, fold_name, step):
    """Process outputs for multi-task mode"""
    # Classification output
    acc1 = accuracy(cls_outputs, cl_label, topk=(1,))
    predicted_labels = torch.argmax(cls_outputs, dim=1)
    predicted_labels = predicted_labels.cpu().item()
    predicted_scores = F.softmax(cls_outputs, dim=1)
    max_value, max_index = torch.max(predicted_scores, dim=1)
    predicted_score = max_value.cpu().item()
    
    # Segmentation output
    threshold = 0.3
    seg_outputs = (seg_outputs > threshold).float()        
    seg_outputs = seg_outputs[0]
    
    # Detection output
    det_val_outputs_all = [det_outputs[0]]
    det_val_targets_all = [det_targets[0]]

    if predicted_labels == 1:
        pre_label = "HGG"
    if predicted_labels == 0:
        pre_label = "LGG"
        
    # Detection metrics
    results_metric = task_manager.matching_batch(
        iou_fn=task_manager.box_utils.box_iou,
        iou_thresholds=det_metric.iou_thresholds,
        pred_boxes=[
            val_data_i[task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
        ],
        pred_classes=[
            val_data_i[task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
        ],
        pred_scores=[
            val_data_i[task_manager.pred_score_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
        ],
        gt_boxes=[val_data_i[task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_targets_all],
        gt_classes=[
            val_data_i[task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_targets_all
        ],
    )
    
    val_det_epoch_metric_dict = det_metric(results_metric)[0]
    mAP_IoU = val_det_epoch_metric_dict['mAP_IoU_0.10_0.50_0.05_MaxDet_10']
    mAR_IoU = val_det_epoch_metric_dict['mAR_IoU_0.10_0.50_0.05_MaxDet_10']
    det_val_epoch_metric = val_det_epoch_metric_dict.values()
    det_val_epoch_metric = sum(det_val_epoch_metric) / len(det_val_epoch_metric)
    det_val_metric = det_val_epoch_metric
    
    # Save results and generate visualizations
    save_multi_task_results(inference_data, det_outputs, inference_img_filenames, 
                           seg_outputs, seg_labels, det_targets, cl_label, pre_label,
                           result_path, fold_name, step)


def process_seg_outputs(inference_data, seg_outputs, seg_labels,
                        inference_img_filenames, result_path, fold_name, step):
    """Process outputs for segmentation task"""
    threshold = 0.3
    seg_outputs = (seg_outputs > threshold).float()
    seg_outputs = seg_outputs[0]
    
    # Save segmentation results
    save_segmentation_results(inference_data, seg_outputs, seg_labels,
                             inference_img_filenames, result_path, fold_name, step)


def process_det_outputs(inference_data, det_outputs, det_targets, task_manager, det_metric,
                        inference_img_filenames, result_path, fold_name, step):
    """Process outputs for detection task"""
    det_val_outputs_all = [det_outputs[0]]
    det_val_targets_all = [det_targets[0]]
    
    # Detection metrics
    results_metric = task_manager.matching_batch(
        iou_fn=task_manager.box_utils.box_iou,
        iou_thresholds=det_metric.iou_thresholds,
        pred_boxes=[
            val_data_i[task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
        ],
        pred_classes=[
            val_data_i[task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
        ],
        pred_scores=[
            val_data_i[task_manager.pred_score_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
        ],
        gt_boxes=[val_data_i[task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_targets_all],
        gt_classes=[
            val_data_i[task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_targets_all
        ],
    )
    
    val_det_epoch_metric_dict = det_metric(results_metric)[0]
    mAP_IoU = val_det_epoch_metric_dict['mAP_IoU_0.10_0.50_0.05_MaxDet_10']
    mAR_IoU = val_det_epoch_metric_dict['mAR_IoU_0.10_0.50_0.05_MaxDet_10']
    
    # Save detection results
    save_detection_results(inference_data, det_outputs, det_targets,
                          inference_img_filenames, result_path, fold_name, step,
                          mAP_IoU, mAR_IoU)

def process_cls_outputs(inference_data, cls_outputs, cl_label,
                        inference_img_filenames, result_path, fold_name, step):
    """Process outputs for classification task"""
    acc1 = accuracy(cls_outputs, cl_label, topk=(1,))
    predicted_labels = torch.argmax(cls_outputs, dim=1)
    predicted_labels = predicted_labels.cpu().item()
    
    if predicted_labels == 1:
        pre_label = "HGG"
    if predicted_labels == 0:
        pre_label = "LGG"
    
    # Save classification results
    save_classification_results(inference_data, cls_outputs, cl_label, pre_label,
                               inference_img_filenames, result_path, fold_name, step)


def save_multi_task_results(inference_data, det_outputs, inference_img_filenames, 
                           seg_outputs, seg_labels, det_targets, cl_label, pre_label,
                           result_path, fold_name, step):
    """Save results and visualizations for multi-task mode"""
    # Visualize and save multi-task results: segmentation + detection + classification
    image = inference_data["image"][0]
    # Detection box
    box = det_outputs[0]["box"].cpu().numpy()
    box_label = det_outputs[0]["box_label"]
    # Segmentation mask
    seg = seg_outputs.cpu().numpy()
    # Classification label
    label_text = f"{pre_label}"
    
    # Ground truth data
    seg_gt = seg_labels[0].cpu().numpy()  # Ground truth segmentation
    gt_box = det_targets[0]["box"].cpu().numpy()  # Ground truth box
    gt_box_label = det_targets[0]["box_label"]
    gt_cl_label = cl_label.cpu().item()  # Ground truth classification
    if gt_cl_label == 1:
        gt_label_text = "HGG"
    else:
        gt_label_text = "LGG"

    # Show one slice (axial)
    slice_num = 60
    # Create subplot with 3 columns: original, prediction, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis('off')
    
    # Prediction (segmentation + detection + classification)
    ax2.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    # Draw detection box
    x_min, y_min, z_min, x_max, y_max, z_max = box[0]
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((y_min, x_min), height, width, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)
    # Dynamically place the label tightly outside the bounding box, never outside the image
    img_h, img_w = image.shape[1], image.shape[2]
    # Candidates: above, below, left, right of the box (centered on each side)
    pred_candidates = [
        (y_min + (y_max - y_min) / 2, max(x_min - 5, 0), 'bottom', 'center'),  # above
        (y_min + (y_max - y_min) / 2, min(x_max + 5, img_h - 1), 'top', 'center'),  # below
        (max(y_min - 5, 0), x_min + (x_max - x_min) / 2, 'center', 'right'),  # left
        (min(y_max + 5, img_w - 1), x_min + (x_max - x_min) / 2, 'center', 'left'),  # right
    ]
    for tx, ty, va, ha in pred_candidates:
        if 0 <= tx < img_w and 0 <= ty < img_h:
            pred_text_x, pred_text_y, va_, ha_ = tx, ty, va, ha
            break
    ax2.text(
        pred_text_x, pred_text_y, f"Pred: {label_text}",
        color='white', fontsize=10, weight='bold',
        bbox=dict(facecolor='red', alpha=1.0, edgecolor='none'),
        verticalalignment=va_, horizontalalignment=ha_
    )
    # Overlay segmentation
    seg_slice = seg[:, :, :, slice_num]
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
    for i in range(3):
        mask = seg_slice[i] > 0
        overlay[mask] = colors[i]
    ax2.imshow(overlay, alpha=0.5)
    ax2.set_title("Prediction", fontsize=12)
    ax2.axis('off')
    
    # Ground truth (segmentation + detection + classification)
    ax3.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    # Draw ground truth detection box
    gt_x_min, gt_y_min, gt_z_min, gt_x_max, gt_y_max, gt_z_max = gt_box[0]
    gt_width = gt_x_max - gt_x_min
    gt_height = gt_y_max - gt_y_min
    gt_rect = patches.Rectangle((gt_y_min, gt_x_min), gt_height, gt_width, linewidth=2, edgecolor='green', facecolor='none')
    ax3.add_patch(gt_rect)
    # Dynamically place the ground truth label tightly outside the bounding box, never outside the image
    gt_candidates = [
        (gt_y_min + (gt_y_max - gt_y_min) / 2, max(gt_x_min - 5, 0), 'bottom', 'center'),  # above
        (gt_y_min + (gt_y_max - gt_y_min) / 2, min(gt_x_max + 5, img_h - 1), 'top', 'center'),  # below
        (max(gt_y_min - 5, 0), gt_x_min + (gt_x_max - gt_x_min) / 2, 'center', 'right'),  # left
        (min(gt_y_max + 5, img_w - 1), gt_x_min + (gt_x_max - gt_x_min) / 2, 'center', 'left'),  # right
    ]
    for tx, ty, va, ha in gt_candidates:
        if 0 <= tx < img_w and 0 <= ty < img_h:
            gt_text_x, gt_text_y, gt_va_, gt_ha_ = tx, ty, va, ha
            break
    ax3.text(
        gt_text_x, gt_text_y, f"GT: {gt_label_text}",
        color='white', fontsize=10, weight='bold',
        bbox=dict(facecolor='green', alpha=1.0, edgecolor='none'),
        verticalalignment=gt_va_, horizontalalignment=gt_ha_
    )
    # Overlay ground truth segmentation
    seg_gt_slice = seg_gt[:, :, :, slice_num]
    overlay_gt = np.zeros((seg_gt_slice.shape[1], seg_gt_slice.shape[2], 3))
    for i in range(3):
        mask = seg_gt_slice[i] > 0
        overlay_gt[mask] = colors[i]
    ax3.imshow(overlay_gt, alpha=0.5)
    ax3.set_title("Ground Truth", fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    file_name = f"{fold_name}_{inference_img_filenames}_multi_task.png"
    full_path = os.path.join(result_path, file_name)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multi-task results for {inference_img_filenames}: {pre_label}, saved to {full_path}")


def save_segmentation_results(inference_data, seg_outputs, seg_labels,
                             inference_img_filenames, result_path, fold_name, step):
    """Save results and visualizations for segmentation task"""
    image = inference_data["image"][0]
    seg = seg_outputs.cpu().numpy()
    seg_gt = seg_labels[0].cpu().numpy()  # Ground truth labels
    slice_num = 60
    
    # Create subplot with 3 columns: original, prediction, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis('off')
    
    # Prediction
    ax2.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    seg_slice = seg[:, :, :, slice_num]
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
    for i in range(3):
        mask = seg_slice[i] > 0
        overlay[mask] = colors[i]
    ax2.imshow(overlay, alpha=0.5)
    ax2.set_title("Prediction", fontsize=12)
    ax2.axis('off')
    
    # Ground truth
    ax3.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    seg_gt_slice = seg_gt[:, :, :, slice_num]
    overlay_gt = np.zeros((seg_gt_slice.shape[1], seg_gt_slice.shape[2], 3))
    for i in range(3):
        mask = seg_gt_slice[i] > 0
        overlay_gt[mask] = colors[i]
    ax3.imshow(overlay_gt, alpha=0.5)
    ax3.set_title("Ground Truth", fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    file_name = f"{fold_name}_{inference_img_filenames}_seg.png"
    full_path = os.path.join(result_path, file_name)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Segmentation results for {inference_img_filenames}, saved to {full_path}")


def save_detection_results(inference_data, det_outputs, det_targets,
                          inference_img_filenames, result_path, fold_name, step,
                          mAP_IoU, mAR_IoU):
    """Save results and visualizations for detection task"""
    image = inference_data["image"][0]
    box = det_outputs[0]["box"].cpu().numpy()
    box_label = det_outputs[0]["box_label"]
    gt_box = det_targets[0]["box"].cpu().numpy()  # Ground truth box
    gt_box_label = det_targets[0]["box_label"]
    slice_num = 60
    
    # Create subplot with 3 columns: original, prediction, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis('off')
    
    # Prediction
    ax2.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    x_min, y_min, z_min, x_max, y_max, z_max = box[0]
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((y_min, x_min), height, width, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)
    ax2.axis('off')
    
    # Ground truth
    ax3.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
    gt_x_min, gt_y_min, gt_z_min, gt_x_max, gt_y_max, gt_z_max = gt_box[0]
    gt_width = gt_x_max - gt_x_min
    gt_height = gt_y_max - gt_y_min
    gt_rect = patches.Rectangle((gt_y_min, gt_x_min), gt_height, gt_width, linewidth=2, edgecolor='green', facecolor='none')
    ax3.add_patch(gt_rect)
    ax3.axis('off')
    
    plt.tight_layout()
    file_name = f"{fold_name}_{inference_img_filenames}_det.png"
    full_path = os.path.join(result_path, file_name)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Detection results for {inference_img_filenames}: mAP={mAP_IoU:.4f}, mAR={mAR_IoU:.4f}, saved to {full_path}")


def save_classification_results(inference_data, cls_outputs, cl_label, pre_label,
                               inference_img_filenames, result_path, fold_name, step):
    """Save results for classification task"""
    acc1 = accuracy(cls_outputs, cl_label, topk=(1,))
    print(f"Classification results for {inference_img_filenames}: {pre_label}, Acc={acc1[0].item():.4f}")

