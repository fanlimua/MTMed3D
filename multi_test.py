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
from openpyxl import load_workbook


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

def load_pre_trained_weights(root_dir, fold_name, model):  
    # Load pre-trained models
    root_dir = os.path.join(root_dir, fold_name) 
    files = os.listdir(root_dir)
    print("Loading Pre-trained Model.")
    for file in files:
        if "seg" in file:
            weights = torch.load(os.path.join(root_dir, file))
            encoder_weights = {}
            for key, value in weights.items():
                if key.startswith("swinViT"):  
                    new_key = key.replace("swinViT.", "") 
                    encoder_weights[new_key] = value
            model.encoder.load_state_dict(encoder_weights)
            print("MultiSwin Shared Encoder has been loaded successfully.")
            
            seg_dec_weights = {}
            for key, value in weights.items():
                if key.startswith("encoder") or key.startswith("decoder") or key.startswith("out"):
                    seg_dec_weights[key] = value
            model.seg_decoder.load_state_dict(seg_dec_weights)
            print("MultiSwin Seg Decoder has been loaded successfully.")
        elif "cls" in file:
            cls_weights = torch.load(os.path.join(root_dir, file))
            cls_dec_weights = {}
            for key, value in cls_weights.items():
                if key.startswith("head"):  
                    cls_dec_weights[key] = value
            model.cls_decoder.load_state_dict(cls_dec_weights)
            print("MultiSwin Cls Decoder has been loaded successfully.")
        elif "det" in file:
            det_weights = torch.load(os.path.join(root_dir, file))
            det_dec_weights = {}
            for key, value in det_weights.items():
                if key.startswith("feature_extractor.fpn"):
                    new_key = key.replace("feature_extractor.fpn", "fpn") 
                    det_dec_weights[new_key] = value
                if key.startswith("classification_head") or key.startswith("regression_head"):
                    det_dec_weights[key] = value
            model.det_decoder.load_state_dict(det_dec_weights)
            print("MultiSwin Det Decoder has been loaded successfully.")

def multi_testing(root_dir, result_dir, test_loader, fold_name, model, task_manager, multi_opt, iou_list=[0.1, 0.5]):

    # Load pre-trained models
    load_pre_trained_weights(root_dir, fold_name, model)
    
    # Metrics
    mean_dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    hd95_metric_batch = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    det_metric = COCOMetric(classes=["tumor"], iou_list=iou_list, max_detection=[20]) 
    
    results_dict = {"validation": []}
    model.eval()
    all_labels = []
    all_preds = []
    det_val_outputs_all = []
    det_val_targets_all = []

    with torch.no_grad():
        start_time = time.time()
        step = 0
        task_manager.multi_setting(model, "init", multi_opt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for inference_data in test_loader:
            inference_data_image = inference_data["image"]
            file_path = inference_data_image.meta["filename_or_obj"]
            file_path = str(file_path)
            cleaned_path = file_path.replace("[", "").replace("]", "").replace("'", "")
            inference_img_filenames = cleaned_path.split("/")[-1]
            file_name_parts = inference_img_filenames.split('.')
            inference_img_filenames = file_name_parts[0]

            step += 1
            print(f"Processing image {step}")

            inference_inputs = inference_data["image"].to(device) 
            det_targets = [dict(
                                        box_label = inference_data["box_label"][0].to(device),
                                        box = inference_data["box"][0].to(device),
                                    )
                        ]
            seg_labels = inference_data["seg_label"].to(device) 
            cl_label = inference_data["cls_label"][0].to(device) 
            cl_label_one_hot = torch.nn.functional.one_hot(torch.as_tensor(cl_label), num_classes=2).float()

            seg_loss, cls_loss, det_loss, seg_outputs, cls_outputs, det_outputs = task_manager.multi_setting(
                    model, 
                    "execute",
                    multi_opt,
                    inference_inputs, 
                    seg_labels,
                    cl_label_one_hot,
                    det_targets,
                    step
                )
            
            #Classification output
            acc1 = accuracy(cls_outputs, cl_label, topk=(1,))
            predicted_labels = torch.argmax(cls_outputs, dim=1)
            predicted_labels = predicted_labels.cpu().item()
            predicted_scores = F.softmax(cls_outputs, dim=1)
            max_value, max_index = torch.max(predicted_scores, dim=1)
            predicted_score = max_value.cpu().item()
            all_labels.extend(cl_label.cpu().numpy())
            all_preds.append(predicted_labels)
            
            #Segmentation output
            threshold = 0.3
            seg_outputs = (seg_outputs > threshold).float()        
            mean_dice_metric(y_pred=seg_outputs, y=seg_labels)
            dice_metric(y_pred=seg_outputs, y=seg_labels)
            hd95_metric(y_pred=seg_outputs, y=seg_labels)
            hd95_metric_batch(y_pred=seg_outputs, y=seg_labels)
            seg_outputs = seg_outputs[0]
            metric_mean = mean_dice_metric.aggregate().item()
            metric_batch = dice_metric.aggregate()
            
            #Detection output
            det_val_outputs_all += det_outputs
            det_val_targets_all.append(det_targets[0])
                    
        # Epoch cls output
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        acc, sensitivity, specificity = sensitivity_specificity(all_preds, all_labels) 
        
        #Epoch seg metric
        metric_mean = mean_dice_metric.aggregate().item()
        metric_batch = dice_metric.aggregate()
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()
        mean_dice_metric.reset()
        dice_metric.reset()

        hd95_metric_mean = hd95_metric.aggregate().item()
        hd95_metric_batch = hd95_metric_batch.aggregate()
        hd95_tc = hd95_metric_batch[0].item()
        hd95_wt = hd95_metric_batch[1].item()
        hd95_et = hd95_metric_batch[2].item()
        hd95_metric.reset()
        # hd95_metric_batch.reset()
        
        # Epoch Det output
        torch.cuda.empty_cache()
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
        # del inference_inputs
        # update inference_data for post transform
        val_det_epoch_metric_dict = det_metric(results_metric)[0]
        mAP_IoU = val_det_epoch_metric_dict.get('mAP_IoU_0.10_0.50_0.05_MaxDet_10')
        mAR_IoU = val_det_epoch_metric_dict.get('mAR_IoU_0.10_0.50_0.05_MaxDet_10')
        
        if mAP_IoU is None:
            for key in val_det_epoch_metric_dict:
                if key.startswith("mAP_IoU_") and not key.startswith("tumor_"):
                    mAP_IoU = val_det_epoch_metric_dict[key]
                    break

        if mAR_IoU is None:
            for key in val_det_epoch_metric_dict:
                if key.startswith("mAR_IoU_") and not key.startswith("tumor_"):
                    mAR_IoU = val_det_epoch_metric_dict[key]
                    break
                
        det_val_epoch_metric = val_det_epoch_metric_dict.values()
        det_val_epoch_metric = sum(det_val_epoch_metric) / len(det_val_epoch_metric)
        det_val_metric = det_val_epoch_metric
        det_metric_list = [(key, round(value, 4)) for key, value in val_det_epoch_metric_dict.items()]
        
        data = [metric_wt, metric_tc, metric_et, hd95_wt, hd95_tc, hd95_et, acc, sensitivity, specificity, mAP_IoU, mAR_IoU]
        data_rounded = [[round(value, 4) for value in data]]  
        return data_rounded, det_metric_list
    
def multi_vis(root_dir, result_dir, test_loader, fold_name, model, task_manager, multi_opt):
    # Metrics
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    hd95_metric_batch = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    mean_dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    det_metric = COCOMetric(classes=["tumor"], iou_list=[0.1], max_detection=[10])
    
    # Load pre-trained models
    load_pre_trained_weights(root_dir, fold_name, model)
    
    results_dict = {"validation": []}
    model.eval()
    all_labels = []
    all_preds = []
    det_val_outputs_all = []
    det_val_targets_all = []

    with torch.no_grad():
        start_time = time.time()
        step = 0
        task_manager.multi_setting(model, "init", multi_opt)
        for inference_data in test_loader:
            inference_data_image = inference_data["image"]
            file_path = inference_data_image.meta["filename_or_obj"]
            file_path = str(file_path)
            cleaned_path = file_path.replace("[", "").replace("]", "").replace("'", "")
            inference_img_filenames = cleaned_path.split("/")[-1]
            file_name_parts = inference_img_filenames.split('.')
            inference_img_filenames = file_name_parts[0]
            # inference_img_filenames = inference_data["image_meta_dict"]["filename_or_obj"] 
            # print(inference_img_filenames)
            step += 1
            print(f"Processing image {step}")

            inference_inputs = inference_data["image"].to(device) 
            det_targets = [dict(
                                        box_label = inference_data["box_label"][0].to(device),
                                        box = inference_data["box"][0].to(device),
                                    )
                        ]
            seg_labels = inference_data["seg_label"].to(device) 
            cl_label = inference_data["cls_label"][0].to(device) 
            cl_label_one_hot = torch.nn.functional.one_hot(torch.as_tensor(cl_label), num_classes=2).float()

            seg_loss, cls_loss, det_loss, seg_outputs, cls_outputs, det_outputs = task_manager.multi_setting(
                    model, 
                    "execute",
                    multi_opt,
                    inference_inputs, 
                    seg_labels,
                    cl_label_one_hot,
                    det_targets,
                    step
                )
            
            #Classification output
            acc1 = accuracy(cls_outputs, cl_label, topk=(1,))
            predicted_labels = torch.argmax(cls_outputs, dim=1)
            predicted_labels = predicted_labels.cpu().item()
            predicted_scores = F.softmax(cls_outputs, dim=1)
            max_value, max_index = torch.max(predicted_scores, dim=1)
            predicted_score = max_value.cpu().item()
            all_labels.extend(cl_label.cpu().numpy())
            all_preds.append(predicted_labels)
            
            #Segmentation output
            threshold = 0.3
            seg_outputs = (seg_outputs > threshold).float()        
            mean_dice_metric(y_pred=seg_outputs, y=seg_labels)
            dice_metric(y_pred=seg_outputs, y=seg_labels)
            hd95_metric(y_pred=seg_outputs, y=seg_labels)
            hd95_metric_batch(y_pred=seg_outputs, y=seg_labels)
            seg_outputs = seg_outputs[0]
            metric_mean = mean_dice_metric.aggregate().item()
            metric_batch = dice_metric.aggregate()
            
            #Detection output
            det_val_outputs_all += det_outputs
            det_val_targets_all.append(det_targets[0])

            if predicted_labels == 1:
                pre_label = "HGG"
            if predicted_labels == 0:
                pre_label = "LGG"
                
            #Detection output
            #torch.cuda.empty_cache()
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
            # del inference_inputs
            # update inference_data for post transform
            val_det_epoch_metric_dict = det_metric(results_metric)[0]
            mAP_IoU = val_det_epoch_metric_dict['mAP_IoU_0.10_0.50_0.05_MaxDet_10']
            mAR_IoU = val_det_epoch_metric_dict['mAR_IoU_0.10_0.50_0.05_MaxDet_10']
            det_val_epoch_metric = val_det_epoch_metric_dict.values()
            det_val_epoch_metric = sum(det_val_epoch_metric) / len(det_val_epoch_metric)
            det_val_metric = det_val_epoch_metric
            
            inference_data["pred_box"] = det_outputs[0]["box"].to(torch.float32)
            inference_data["pred_label"] = det_outputs[0]["box_label"]
            inference_data["pred_score"] = det_outputs[0]["box_label_scores"].to(torch.float32)
            # inference_data[i] = det_post_transforms(inference_data_i)
            result = {
                "label": inference_data["pred_label"].cpu().detach().numpy().tolist(),
                "box": inference_data["pred_box"].cpu().detach().numpy().tolist(),
                "score": inference_data["pred_score"].cpu().detach().numpy().tolist(),
            }
            result.update({"image": inference_img_filenames})
            results_dict["validation"].append(result)

            scores = result["score"]
            if len(scores) != 0:
                # Create the saved folder
                date = str(time.strftime("%m_%d_%H_%M", time.localtime()))
                # folder_name = f"test_{date}"
                # result_dir = os.path.join(result_dir, folder_name)
                # if not os.path.exists(result_dir):
                #     os.mkdir(result_dir)
                # else:
                #     print(f"Directory '{result_dir}' already exists, no need to create.")
                
                max_score = np.max(scores)
                max_score_index = np.argmax(scores)
                max_score_label = result["label"][max_score_index]
                max_score_box = result["box"][max_score_index]
                
                coords_values_tensor = det_targets[0]["box"]
                coords_values = coords_values_tensor.cpu().numpy()
                
                l = open(os.path.join(result_dir, f'metric_result_{date}.txt'), 'a+')
                l.write(f"{inference_img_filenames}; Mean Dice: {metric_mean:.4f}; WT: {metric_batch[1].item():.4f}; TC: {metric_batch[0].item():.4f}; ET: {metric_batch[2].item():.4f}; Grading: {pre_label}; Acc: {acc1[0].item()}; Det Mean Metric: {det_val_metric:.4f}; Det mAP: {mAP_IoU:.4f}; Det mAR: {mAR_IoU:.4f}; Det coords: {max_score_box}; GT coords: {coords_values}\n")
                print(f"{inference_img_filenames}; Mean Dice: {metric_mean:.4f}; WT: {metric_batch[1].item():.4f}; TC: {metric_batch[0].item():.4f}; ET: {metric_batch[2].item():.4f}; Grading: {pre_label}; Acc: {acc1[0].item()}; Det Mean Metric: {det_val_metric:.4f}; Det mAP: {mAP_IoU:.4f}; Det mAR: {mAR_IoU:.4f}")
                mean_dice_metric.reset()
                dice_metric.reset()
                seg_labels = inference_data["seg_label"][0].to(device) 

                #Detction results
                if max_score_label == 0:
                    box_label = "Tumor"
                x_min = max_score_box[0]
                y_min = max_score_box[1]
                z_min = max_score_box[2]
                x_max = max_score_box[3]
                y_max = max_score_box[4]
                z_max = max_score_box[5]
                width  = x_max - x_min
                height = y_max - y_min
                depth  = z_max - z_min
                
                #Top
                slice_nums = [45, 50, 55] 
                # slice_nums = [30, 40, 45, 50, 55, 60, 65, 80] 
                # slice_num = 60
                for i, slice_num in enumerate(slice_nums):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                    image = inference_data["image"][0]
                    # ax1.set_title('Axial Plane')
                    ax1.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
                    rect1 = patches.Rectangle((y_min, x_min), height, width, linewidth=2, edgecolor='r', facecolor='none')
                    ax1.add_patch(rect1)
                    # label_text = f"{pre_label}; {predicted_score:.2f}"
                    label_text = f"{pre_label}"
                    
                    text = ax1.text(0, 0, label_text, fontsize=15, weight='bold')
                    renderer = fig.canvas.get_renderer()
                    bbox = text.get_window_extent(renderer=renderer)
                    text_height = bbox.height / fig.dpi * 72  
                    text.remove()  
                    label_y = x_min - text_height - 2  

                    # Calculate the top-right corner outside the bounding box
                    label_x = y_max + 4  
                    label_y = x_min - 4  

                    image_width, image_height = 96, 96  

                    # Check for boundary conditions
                    if label_x > image_width or label_y < 0:  # Out of right or top boundary
                        if label_x > image_width and label_y < 0:  # Both x and y out of bounds
                            label_x = y_min - 4  # Move to the left of the box
                            label_y = x_max + 4  # Move below the box
                        elif label_x > image_width:  # Only x out of bounds
                            label_x = y_min - 4  # Move to the left of the box
                        elif label_y < 0:  # Only y out of bounds
                            label_y = x_max + 4  # Move below the box

                    # Add the text at the calculated position
                    text = ax1.text(
                        label_x, label_y, label_text,
                        color='white', fontsize=10, weight='bold',
                        ha='right' if label_x > y_min else 'left',  
                        va='top' if label_y < x_max else 'bottom',
                        bbox=dict(facecolor='red', alpha=1.0, edgecolor='none') 
                    )

                    seg_slice = seg_outputs[:, :, :, slice_num].detach().cpu().numpy()
                    colors = np.array([
                        [1, 0, 0],  # Red 
                        [0, 1, 0],  # Green 
                        [0, 0, 1],  # Blue 
                    ])
                    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
                    mask = seg_slice[1] > 0
                    overlay[mask] = colors[1]
                    mask = seg_slice[0] > 0
                    overlay[mask] = colors[0]
                    mask = seg_slice[2] > 0
                    overlay[mask] = colors[2]
                    ax1.imshow(overlay, alpha=0.5) 

                    #Side
                    # ax2.set_title('Sagittal Plane')
                    ax2.imshow(image[1, :, slice_num, :].detach().cpu(), cmap="gray")
                    rect2 = patches.Rectangle((z_min, x_min), depth, width, linewidth=2, edgecolor='r', facecolor='none')
                    ax2.add_patch(rect2)
                    seg_slice = seg_outputs[:, :, slice_num, :].detach().cpu().numpy()
                    colors = np.array([
                        [1, 0, 0],  # Red 
                        [0, 1, 0],  # Green 
                        [0, 0, 1],  # Blue 
                    ])
                    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
                    mask = seg_slice[1] > 0
                    overlay[mask] = colors[1]
                    mask = seg_slice[0] > 0
                    overlay[mask] = colors[0]
                    mask = seg_slice[2] > 0
                    overlay[mask] = colors[2]
                    ax2.imshow(overlay, alpha=0.5) 

                    #Front
                    # ax3.set_title('Coronal Plane')
                    ax3.imshow(image[1, slice_num, :, :].detach().cpu(), cmap="gray")
                    rect3 = patches.Rectangle((z_min, y_min), depth, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax3.add_patch(rect3)
                    seg_slice = seg_outputs[:, slice_num, :, :].detach().cpu().numpy()
                    colors = np.array([
                        [1, 0, 0],  # Red 
                        [0, 1, 0],  # Green 
                        [0, 0, 1],  # Blue 
                    ])
                    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
                    mask = seg_slice[1] > 0
                    overlay[mask] = colors[1]
                    mask = seg_slice[0] > 0
                    overlay[mask] = colors[0]
                    mask = seg_slice[2] > 0
                    overlay[mask] = colors[2]
                    ax3.imshow(overlay, alpha=0.5) 
                    
                    file_name_pre = f"{fold_name}_{inference_img_filenames}_{slice_num}_pre.png"
                    full_path_pre = os.path.join(result_dir, file_name_pre)
                    plt.savefig(full_path_pre)
                    print("Output images are saved successfully.")
                    
                    #Ground Truth
                    # slice_num = 60
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                    coords_values_tensor = det_targets[0]["box"]
                    coords_values = coords_values_tensor.cpu().numpy()
                    min_coords = coords_values[0][:3]
                    max_coords = coords_values[0][3:]
                    x_min_gt = min_coords[0]
                    y_min_gt = min_coords[1]
                    z_min_gt = min_coords[2]
                    x_max_gt = max_coords[0]
                    y_max_gt = max_coords[1]
                    z_max_gt = max_coords[2]
                    width_gt  = x_max_gt - x_min_gt
                    height_gt = y_max_gt - y_min_gt
                    depth_gt  = z_max_gt - z_min_gt
                    print(f"Min coordinates: {min_coords}, Max coordinates: {max_coords}.")
                    if cl_label == 0:
                        gt_label = "LGG"
                    if cl_label == 1:
                        gt_label = "HGG"
                    if det_targets[0]["box_label"] == 0:
                        box_label = "tumor"
                    #Top
                    ax1.set_title('Axial Plane')
                    ax1.imshow(image[1, :, :, slice_num].detach().cpu(), cmap="gray")
                    rect1 = patches.Rectangle((y_min_gt, x_min_gt), height_gt, width_gt, linewidth=2, edgecolor='r', facecolor='none')
                    ax1.add_patch(rect1)
                    
                    label_text = f"{gt_label}"
                    text = ax1.text(0, 0, label_text, fontsize=15, weight='bold')
                    renderer = fig.canvas.get_renderer()
                    bbox = text.get_window_extent(renderer=renderer)
                    text_height = bbox.height / fig.dpi * 72  
                    text.remove()  
                    label_gt_y = x_min_gt - text_height - 2
                    
                    # Calculate the top-right corner outside the bounding box
                    label_gt_x = y_max_gt + 4  
                    label_gt_y = x_min_gt - 4  

                    image_width, image_height = 96, 96  

                    # Check for boundary conditions
                    if label_gt_x > image_width or label_gt_y < 0:  # Out of right or top boundary
                        if label_gt_x > image_width and label_gt_y < 0:  # Both x and y out of bounds
                            label_gt_x = y_min_gt - 4  # Move to the left of the box
                            label_gt_y = x_max_gt + 4  # Move below the box
                        elif label_gt_x > image_width:  # Only x out of bounds
                            label_gt_x = y_min_gt - 4  # Move to the left of the box
                        elif label_gt_y < 0:  # Only y out of bounds
                            label_gt_y = x_max_gt + 4  # Move below the box

                    # Add the text at the calculated position
                    text = ax1.text(
                        label_gt_x, label_gt_y, label_text,
                        color='white', fontsize=10, weight='bold',
                        ha='right' if label_gt_x > y_min_gt else 'left',  
                        va='top' if label_gt_y < x_max_gt else 'bottom',
                        bbox=dict(facecolor='red', alpha=1.0, edgecolor='none') 
                    )
                    
                    seg_slice = seg_labels[:, :, :, slice_num].detach().cpu().numpy()
                    colors = np.array([
                        [1, 0, 0],  # Red 
                        [0, 1, 0],  # Green 
                        [0, 0, 1],  # Blue 
                    ])
                    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
                    mask = seg_slice[1] > 0
                    overlay[mask] = colors[1]
                    mask = seg_slice[0] > 0
                    overlay[mask] = colors[0]
                    mask = seg_slice[2] > 0
                    overlay[mask] = colors[2]
                    ax1.imshow(overlay, alpha=0.5) 

                    #Side
                    ax2.set_title('Sagittal Plane')
                    ax2.imshow(image[1, :, slice_num, :].detach().cpu(), cmap="gray")
                    rect2 = patches.Rectangle((z_min_gt, x_min_gt), depth_gt, width_gt, linewidth=2, edgecolor='r', facecolor='none')
                    ax2.add_patch(rect2)
                    seg_slice = seg_labels[:, :, slice_num, :].detach().cpu().numpy()
                    colors = np.array([
                        [1, 0, 0],  # Red 
                        [0, 1, 0],  # Green 
                        [0, 0, 1],  # Blue 
                    ])
                    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
                    mask = seg_slice[1] > 0
                    overlay[mask] = colors[1]
                    mask = seg_slice[0] > 0
                    overlay[mask] = colors[0]
                    mask = seg_slice[2] > 0
                    overlay[mask] = colors[2]
                    ax2.imshow(overlay, alpha=0.5) 

                    #Front
                    ax3.set_title('Coronal Plane')
                    ax3.imshow(image[1, slice_num, :, :].detach().cpu(), cmap="gray")
                    rect3 = patches.Rectangle((z_min_gt, y_min_gt), depth_gt, height_gt, linewidth=2, edgecolor='r', facecolor='none')
                    ax3.add_patch(rect3)
                    seg_slice = seg_labels[:, slice_num, :, :].detach().cpu().numpy()
                    colors = np.array([
                        [1, 0, 0],  # Red Tumor Core
                        [0, 1, 0],  # Green Whold Tumor
                        [0, 0, 1],  # Blue Enhanced Tumor
                    ])
                    overlay = np.zeros((seg_slice.shape[1], seg_slice.shape[2], 3))
                    mask = seg_slice[1] > 0
                    overlay[mask] = colors[1]
                    mask = seg_slice[0] > 0
                    overlay[mask] = colors[0]
                    mask = seg_slice[2] > 0
                    overlay[mask] = colors[2]
                    ax3.imshow(overlay, alpha=0.5) 

                    file_name = f"{fold_name}_{inference_img_filenames}_{slice_num}_gt.png"
                    full_path = os.path.join(result_dir, file_name)
                    plt.savefig(full_path)
                    torch.cuda.empty_cache()
                    plt.clf()
                    print("Output images are saved successfully.")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Multi-task model testing.")
    parser.add_argument('--json_file', type=str, default='./utils/data_annotation_2018.json')
    parser.add_argument('--data_dir', type=str, default='../dataset/BraTS/imageTr2018')
    parser.add_argument('--seg_label_dir', type=str, default='../dataset/BraTS/labelTr2018')
    parser.add_argument("--root_dir", type=str, default="../Medical-image-analysis/models/Dense_PANet_GradNorm", help="Directory containing model weights.")
    parser.add_argument("--result_dir", type=str, default="./result/test_multi", help="Directory for saving results.")
    parser.add_argument("--iou_list", type=str, default=[0.1, 0.5], help="Comma-separated IoU thresholds")
    parser.add_argument("--mode", type=str, default="testing", help="'testing' for validating model performance metrics, or 'efficiency' for evaluating resource usage and runtime performance, or 'vis' for visulization.")
    parser.add_argument("--multi_opt", type=str, default="GradNorm", help="Traning Optimizer")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task_manager = ini_models(device)
    
    #BraST2018
    full_files = load_decathlon_datalist(
            args.json_file,
            is_segmentation=True,
            data_list_key="training",
            base_dir=args.data_dir,
            seg_label_dir=args.seg_label_dir,
        )
    
    if args.mode == "testing":
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = kf.split(full_files)
        results = []
        det_metric_list = []
        
        for i, (train_index, val_index) in enumerate(folds):
            fold_name = f"fold_{i+1}" 
            train_files = load_data_indices(full_files, train_index)
            val_files = load_data_indices(full_files, val_index)
            
            train_ds = Dataset(
                    data=train_files,
                    transform=train_transforms,
                    )
            val_ds = Dataset(
                    data=val_files,
                    transform=val_transforms,
                )
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
            print(f"Test data fold {i+1} loaded")
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            test_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            print("Start training...")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Testing
            data_rounded, det_metrics = multi_testing(
                                root_dir=args.root_dir,
                                result_dir=args.result_dir,
                                test_loader=val_loader,
                                fold_name=fold_name,
                                model = model,
                                task_manager = task_manager,
                                multi_opt = args.multi_opt,
                                iou_list=args.iou_list
                            )
            
            results.append(data_rounded)
            det_metric_list.append(det_metrics)
        
        results = [item[0] for item in results]
        folds = [f"fold{fold}" for fold in range(1, len(results) + 1)]

        headers = ["Fold", "Dice_WT", "Dice_TC", "Dice_EH", "HD_WT", "HD_TC", "HD_ET", 
                "Accuracy", "Sen(pos)", "Spe(neg)", "mAP_IoU", "mAR_IoU"]

        # Modify the list comprehension to unpack each result into individual columns
        results_with_folds = [[fold] + result for fold, result in zip(folds, results)]

        # Print the table using tabulate
        print(tabulate(results_with_folds, headers=headers, tablefmt="grid"))
        
        det_metrics_dicts = [dict(det) for det in det_metric_list]
        metric_names = sorted({k for d in det_metrics_dicts for k in d})
        transpose_table = []
        for metric in metric_names:
            row = [metric] + [round(det.get(metric, None), 4) for det in det_metrics_dicts]
            transpose_table.append(row)
        transpose_headers = ["Metric"] + [f"fold{i+1}" for i in range(len(det_metrics_dicts))]
        print("\nDetection Metrics")
        print(tabulate(transpose_table, headers=transpose_headers, tablefmt="grid"))
        
        df1 = pd.DataFrame(results_with_folds, columns=headers)
        df2 = pd.DataFrame(transpose_table, columns=transpose_headers)
        folder_name = os.path.basename(os.path.normpath(args.root_dir))
        excel_filename = os.path.join("result/test_multi", f"{folder_name}_results.xlsx")
        with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
            df1.to_excel(writer, index=False, sheet_name="Results", startrow=0)
            df2.to_excel(writer, index=False, sheet_name="Results", startrow=len(df1) + 2)

    
    elif args.mode == "vis":
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = kf.split(full_files)
        results = []
        
        for i, (train_index, val_index) in enumerate(folds):
            if i != 1:  # Skip all folds except fold2 (i == 1)
                continue
            fold_name = f"fold_{i+1}" 
            train_files = load_data_indices(full_files, train_index)
            val_files = load_data_indices(full_files, val_index)
            
            train_ds = Dataset(
                    data=train_files,
                    transform=train_transforms,
                    )
            val_ds = Dataset(
                    data=val_files,
                    transform=val_transforms,
                )
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
            print(f"Test data fold {i+1} loaded")
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            test_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            print("Start training...")
            
            # Testing
            data_rounded = multi_vis(
                                root_dir=args.root_dir,
                                result_dir=args.result_dir,
                                test_loader=val_loader,
                                fold_name=fold_name,
                                model = model,
                                task_manager = task_manager,
                                multi_opt = args.multi_opt
                            )
          
    elif args.mode == "efficiency":
        load_pre_trained_weights("/home/fan/project/Medical-image-analysis/models/Dense_PANet_GradNorm/", "fold_1", model)
        input_size = 96
        results = []
        
        macs, flops = calculate_macs_with_ptflops(model, input_size)
        num_params = count_parameters(model)
        inference_time = measure_inference_time(model, "MultiSwin", input_size, device, 'multi')
        model_size = get_model_size_from_pytorch(model)
        
        results.append([
            "MultiSwin",
            macs,
            flops,
            num_params,
            inference_time,
            model_size
        ])
            
        headers = ["Model Name", "MACs", "FLOPs", "Parameters", "Inference Time (s)", "Model Size (MB)"]
        table = tabulate(results, headers=headers, tablefmt="grid", floatfmt=".4f")
        print(table)
            
        
      
    