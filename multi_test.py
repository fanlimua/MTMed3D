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
from utils.model_utils import *
import gc

def multi_testing(config, use_integrated_model, root_dir, train_loader, val_loader, test_loader, fold_name, model):
    # Load pre-trained models
    if use_integrated_model:
        load_pre_trained_weights_multiswin(root_dir, fold_name, model, config.task)
    else:
        load_pre_trained_weights(root_dir, fold_name, model, config.task)
    
    model.to(device)
    solver = Trainer(config, train_loader, val_loader, test_loader, root_dir)
    if config.task == "multi":
        solver.task_manager.multi_setting(model, "init", config.multi_opt)
        # results = solver.validate(model, config.task, 1, val_loader, device)
    elif config.task == "segmentation":
        solver.task_manager.seg_setting(model, action = "init")
    elif config.task == "detection":
        solver.task_manager.det_setting(model, action = "init")
    elif config.task == "classification":
        solver.task_manager.cls_setting(model, action = "init")
    
    results = solver.validate(model, config.task, 1, val_loader, device)
    return results
    
def multi_vis(config, use_integrated_model, root_dir, result_path, test_loader, fold_name, model, task_manager, multi_opt):
    det_metric = COCOMetric(classes=["tumor"], iou_list=[0.1], max_detection=[10])
    
    # Load pre-trained models
    if use_integrated_model:
        load_pre_trained_weights_multiswin(root_dir, fold_name, model, config.task)
    else:
        load_pre_trained_weights(root_dir, fold_name, model, config.task)
    
    model.to(device)
    model.eval()

    if config.vismode == "basic": 
        with torch.no_grad():
            start_time = time.time()
            step = 0
            
            # Initialize according to task type
            if config.task == "multi":
                task_manager.multi_setting(model, "init", multi_opt)
            elif config.task == "segmentation":
                task_manager.seg_setting(model, "init")
            elif config.task == "detection":
                task_manager.det_setting(model, "init")
            elif config.task == "classification":
                task_manager.cls_setting(model, "init")
            
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
                
                # Run inference according to task type
                if config.task == "multi":
                    # Multi-task mode
                    det_targets = [dict(
                        box_label = inference_data["box_label"][0].to(device),
                        box = inference_data["box"][0].to(device),
                    )]
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
                    
                    # Process multi-task outputs
                    process_multi_task_outputs(inference_data, seg_outputs, cls_outputs, det_outputs, 
                                             seg_labels, cl_label, det_targets, task_manager, det_metric,
                                             inference_img_filenames, result_path, fold_name, step)
                    
                elif config.task == "segmentation":
                    # Segmentation only
                    seg_labels = inference_data["seg_label"].to(device)
                    seg_loss, seg_outputs = task_manager.seg_setting(
                        model, 
                        "execute",
                        inference_inputs, 
                        seg_labels
                    )
                    
                    # Process segmentation outputs
                    process_seg_outputs(inference_data, seg_outputs, seg_labels,
                                        inference_img_filenames, result_path, fold_name, step)
                    
                elif config.task == "detection":
                    # Detection only
                    det_targets = [dict(
                        box_label = inference_data["box_label"][0].to(device),
                        box = inference_data["box"][0].to(device),
                    )]
                    det_outputs, det_loss, det_losses = task_manager.det_setting(
                        model, 
                        "execute",
                        inference_inputs, 
                        det_targets
                    )
                    
                    # Process detection outputs
                    process_det_outputs(inference_data, det_outputs, det_targets, task_manager, det_metric,
                                        inference_img_filenames, result_path, fold_name, step)
                    
                elif config.task == "classification":
                    # Classification only
                    cl_label = inference_data["cls_label"][0].to(device)
                    cl_label_one_hot = torch.nn.functional.one_hot(torch.as_tensor(cl_label), num_classes=2).float()
                    cls_loss, cls_outputs = task_manager.cls_setting(
                        model, 
                        "execute",
                        inference_inputs, 
                        cl_label_one_hot
                    )
                    
                    # Process classification outputs
                    process_cls_outputs(inference_data, cls_outputs, cl_label,
                                        inference_img_filenames, result_path, fold_name, step)
                        
    elif config.vismode == "grad_cam":
        # Grad-CAM visualization mode
        for inference_data in test_loader:
            inference_data_image = inference_data["image"]
            file_path = inference_data_image.meta["filename_or_obj"]
            file_path = str(file_path)
            cleaned_path = file_path.replace("[", "").replace("]", "").replace("'", "")
            inference_img_filenames = cleaned_path.split("/")[-1]
            file_name_parts = inference_img_filenames.split('.')
            inference_img_filenames = file_name_parts[0]

            inference_inputs = inference_data["image"].to(device) 
            det_targets = [dict(
                box_label = inference_data["box_label"][0].to(device),
                box = inference_data["box"][0].to(device),
            )]
            seg_labels = inference_data["seg_label"].to(device) 
            cl_label = inference_data["cls_label"][0].to(device) 
            cl_label_one_hot = torch.nn.functional.one_hot(torch.as_tensor(cl_label), num_classes=2).float()
            
            target_layer_obj = eval(f"model.{config.target_layers}")
            if "seg" in config.target_layers:
                model_task = "segmentation"
            elif "cls" in config.target_layers:
                model_task = "classification"
            elif "det" in config.target_layers:
                model_task = "detection"
                
            if model_task == "classification":
                wrapped_model = ModelWrapper(model, task=model_task)
                target = ClassifierOutputTarget(config.vis_target)
                cam = GradCAM(model=wrapped_model, target_layers=[target_layer_obj])
                grayscale_cam = cam(input_tensor=inference_inputs, targets=[target])
                upscaled_heatmap = np.uint8(255 * grayscale_cam)

                slice_num = 30
                inputs_show = inference_inputs[0].cpu()
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(inputs_show[0, :, :, slice_num],cmap="gray")
               
                ax2.imshow(inputs_show[0, :, :, slice_num].detach().cpu(), cmap="gray")
                seg_slice = seg_labels[0, :, :, :, slice_num].detach().cpu().numpy()
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
                
                img0 = ax3.imshow(inputs_show[0, :, :, slice_num],cmap="gray")
                ax3.imshow(upscaled_heatmap[..., slice_num].squeeze(),
                                    cmap='jet', alpha=0.3, extent=img0.get_extent())
                heatmap_output_path = os.path.join(result_path, f'{inference_img_filenames}_target{config.vis_target}.png')
                plt.savefig(heatmap_output_path)
                plt.close(fig)
                print(f"Successfully saved image {inference_img_filenames} heatmap.")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Multi-task model testing.")
    parser.add_argument('--json_file', type=str, default='./utils/data_annotation_2018.json')
    parser.add_argument('--data_dir', type=str, default='../dataset/BraTS/imageTr2018')
    parser.add_argument('--seg_label_dir', type=str, default='../dataset/BraTS/labelTr2018')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_k_fold', default=False)
    parser.add_argument("--use_integrated_model", type=bool, default=False, 
                        help="If set, load the integrated MultiSwin model; otherwise, load three separate single-task models.")
    parser.add_argument('--task', type=str, default='classification', help="Specify the type of training. Options: segmentation(single-task), detection(single-task), classification(single-task), multi(multi-task).")
    parser.add_argument("--root_dir", type=str, 
                        default="/home/fan/project/Medical-image-analysis/models/Dense_PANet_GradNorm/fold_1",
                        help="Directory containing model weights.")
    parser.add_argument("--result_path", type=str, 
                        default="/home/fan/project/MultiSwin/result/test_multi/vis/vis_dir",
                        help="Directory for saving results.")
    parser.add_argument("--mode", type=str, default="vis", 
                        choices=["testing", "efficiency", "vis"],
                        help="'testing' for validating model performance metrics,"
                        "or 'efficiency' for evaluating resource usage and runtime performance, or 'vis' for visulization.")
    parser.add_argument("--vismode", type=str, default="grad_cam", 
                        choices=["basic", "grad_cam"],
                        help="Specify the visualization mode to use.")
    parser.add_argument("--vis_target", type=int, default=1, help="Specify the target we want to generate the Class Activation Maps for. If targets is None, the highest scoring category.")
    parser.add_argument("--target_layers", type=str, default="cls_decoder.head.features[-2]", help="Target layer for Grad-CAM.")
    parser.add_argument("--iou_list", type=str, default=[0.1], help="Comma-separated IoU thresholds")
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
        if args.use_k_fold:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = kf.split(full_files)
            # results = []
            # det_metric_list = []
            summary_headers = ["Fold", "Dice_WT", "Dice_TC", "Dice_EH", "HD_WT", "HD_TC", "HD_ET",
                   "Accuracy", "Sen(pos)", "Spe(neg)", "mAP_IoU", "mAR_IoU"]
            summary_table = []
            detection_metrics_per_fold = []
            
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
                fold_results = multi_testing(
                                    config=args,
                                    use_integrated_model=args.use_integrated_model,
                                    root_dir=args.root_dir,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    test_loader=val_loader,
                                    fold_name=fold_name,
                                    model = model
                                )

                result_row, det_metric_dict = wrap_validation_result(args.task, fold_results)
                result_row["Fold"] = f"{fold_name}"
                summary_table.append(result_row)

                # Store detection metric dictionary
                detection_metrics_per_fold.append(dict(det_metric_dict))
            
            all_detection_keys = sorted({k for d in detection_metrics_per_fold for k in d})
            transpose_table = []
            for metric in all_detection_keys:
                row = [metric] + [
                    round(det.get(metric, None), 4) if metric in det else None
                    for det in detection_metrics_per_fold
                ]
                transpose_table.append(row)

            rows = [[row.get(h, "") for h in summary_headers] for row in summary_table]
            print("\nOverall Metrics:")
            print(tabulate(rows, headers=summary_headers, tablefmt="grid"))
            
            transpose_headers = ["Metric"] + [f"fold{i+1}" for i in range(len(detection_metrics_per_fold))]
            print("\nDetection Metrics:")
            print(tabulate(transpose_table, headers=transpose_headers, tablefmt="grid"))
            
            # Create DataFrames
            df_summary = pd.DataFrame(summary_table)[summary_headers]
            df_detection = pd.DataFrame(transpose_table, columns=transpose_headers)

            # Create output path
            folder_name = os.path.basename(os.path.normpath(args.root_dir))
            output_path = os.path.join(args.result_path, f"{folder_name}_results.xlsx")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save both tables to one Excel file
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df_summary.to_excel(writer, index=False, sheet_name="Results", startrow=0)
                df_detection.to_excel(writer, index=False, sheet_name="Results", startrow=len(df_summary) + 2)
                
        else:
            train_files = full_files[:int(0.8 * len(full_files))]
            val_files = full_files[int(0.8 * len(full_files)):]
            train_ds = Dataset(
                    data=train_files,
                    transform=train_transforms,
                    )
            val_ds = Dataset(
                    data=val_files,
                    transform=val_transforms,
                )
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
            print("Train data loaded")
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            test_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Testing
            fold_results = multi_testing(
                                config=args,
                                use_integrated_model=args.use_integrated_model,
                                root_dir=args.root_dir,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                test_loader=val_loader,
                                fold_name="",
                                model = model
                            )
            result_row, det_metric_dict = wrap_validation_result(args.task, fold_results)
            
            # Overall table
            summary_headers = list(result_row.keys())
            summary_values = [result_row[h] for h in summary_headers]
            print("\nOverall Metrics:")
            print(tabulate([summary_values], headers=summary_headers, tablefmt="grid"))
            
            # Detection table
            # print("\nDetection Metrics:")
            # print(tabulate(det_metric_dict, headers=["Metric", "Value"], tablefmt="grid"))
            
    elif args.mode == "vis":
        os.makedirs(args.result_path, exist_ok=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = kf.split(full_files)
        results = []
        
        if args.use_k_fold:
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
                                config=args,
                                use_integrated_model=args.use_integrated_model,
                                root_dir=args.root_dir,
                                result_path=args.result_path,
                                test_loader=val_loader,
                                fold_name="",
                                model = model,
                                task_manager = task_manager,
                                multi_opt = args.multi_opt
                            )
                
        else:
            train_files = full_files[:int(0.8 * len(full_files))]
            val_files = full_files[int(0.8 * len(full_files)):]
            train_ds = Dataset(
                    data=train_files,
                    transform=train_transforms,
                    )
            val_ds = Dataset(
                    data=val_files,
                    transform=val_transforms,
                )
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
            print("Train data loaded")
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            test_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            
            # Testing
            data_rounded = multi_vis(
                                config=args,
                                use_integrated_model=args.use_integrated_model,
                                root_dir=args.root_dir,
                                result_path=args.result_path,
                                test_loader=val_loader,
                                fold_name="",
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
            
        
      
    