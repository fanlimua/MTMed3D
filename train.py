import gc
import os
import numpy as np
import torch
from lib.task_manager import *
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss, DiceCELoss, HausdorffDTLoss
from model.network import MultiSwin
from monai.apps.detection.metrics.coco import COCOMetric
from timm.utils import accuracy, AverageMeter
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose)
from lib.evaluation import sensitivity_specificity


class Trainer(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, result_path):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.root_dir = result_path

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # Path
        # self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.task = config.task
        self.multi_opt = config.multi_opt
        self.iou_list = config.iou_list

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.3)])
        
        #Modles
        self.model = MultiSwin(img_size=(96,96,96), in_channels=4)
        self.model.to(self.device)
        
        # Loss & Optimizer
        self.task_manager = TaskManager()
        
        # Metrics
        self.mean_dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
        self.hd95_metric_batch = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
        self.det_metric = COCOMetric(classes=["tumor"], iou_list=self.iou_list, max_detection=[10])

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def save_checkpoint(self, epoch, model, optimizer, output_dir, filename, is_best=False):
        model_state = model.state_dict()
        checkpoint = {
                'epoch': epoch,
                'state_dict': model_state,
                'optimizer': optimizer.state_dict(),
            }
        torch.save(checkpoint, os.path.join(output_dir, filename))
        # if is_best and 'state_dict' in checkpoint:
        #     torch.save(checkpoint['best_state_dict'],
        #             os.path.join(output_dir, 'model_best.pth'))
        
    def plot_subplots(self, x, y_list, titles, xlabel, ylabel, colors, save_path, figsize=(18, 6)):
        """
        Plots multiple subplots in a single figure.
        """
        plt.figure(figsize=figsize)
        num_plots = len(y_list)
        for i, (y, title, color) in enumerate(zip(y_list, titles, colors), start=1):
            plt.subplot(1, num_plots, i)
            plt.plot(x, y, linewidth=1, linestyle="solid", color=color)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def plot_multiple_curves(self, x, y_list, labels, xlabel, ylabel, title, colors, save_path=None):
        """
        Plots multiple curves on the same figure.
        """
        plt.figure(figsize=(8, 6))
        for y, label, color in zip(y_list, labels, colors):
            plt.plot(x, y, linewidth=1, linestyle="solid", label=label, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    # ====================================== Training ===========================================#

    def train_one_epoch(self, model, task, epoch_index, train_loader, device):
        model.train()
        total_loss_all = 0
        seg_epoch_loss = 0
        det_epoch_loss = 0
        cls_epoch_loss = 0
        epoch_cls_loss = 0
        cl_epoch_loss = 0
        epoch_box_reg_loss = 0
        step = 0
        
        if task == "segmentation":
            self.task_manager.seg_setting(model, action = "init")
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["image"].to(device), batch_data["seg_label"].to(device)
                
                seg_loss = self.task_manager.seg_setting(
                    model, 
                    "execute",
                    inputs, 
                    labels
                )
                seg_epoch_loss += seg_loss.item()
                
                current_lr = self.task_manager.seg_optimizer.param_groups[0]['lr']
                print(f"{epoch_index} - {step}/{len(train_loader)} - Loss: {seg_loss.item()} - Learning rate: {current_lr}")
                torch.cuda.empty_cache()
            
            self.task_manager.seg_lr_scheduler.step()
            seg_epoch_loss /= step
            return seg_epoch_loss
        
        elif task == "detection":
            self.task_manager.det_setting(model, action = "init")
            for batch_data in train_loader:
                step += 1
                inputs = batch_data["image"].to(device) 
                targets = [dict(
                                    box_label = batch_data["box_label"][0].to(device),
                                    box = batch_data["box"][0].to(device),
                                )
                        ]
                
                det_loss, det_losses = self.task_manager.det_setting(
                    model, 
                    "execute",
                    inputs, 
                    targets
                )

                det_epoch_loss += det_loss.detach().item()
                epoch_cls_loss += det_losses[self.task_manager.cls_key].detach().item()
                epoch_box_reg_loss += det_losses[self.task_manager.box_reg_key].detach().item()
                current_lr = self.task_manager.det_optimizer.param_groups[0]['lr']
                print(f"{epoch_index} - {step}/{len(train_loader)} - Loss: {det_loss.item():.4f} - Learning rate: {current_lr}")

            self.task_manager.det_lr_scheduler.step()
            det_epoch_loss /= step
            epoch_cls_loss /= step
            epoch_box_reg_loss /= step
            return det_epoch_loss, epoch_cls_loss, epoch_box_reg_loss
        
        elif task == "classification":
            self.task_manager.cls_setting(model, action = "init")
            loss_meter = AverageMeter()
            cls_train_loss = 0
            for batch_data in train_loader:
                step += 1
                inputs = batch_data["image"].to(device) 
                cl_label = batch_data["cls_label"][0].to(device)
                cl_label = torch.nn.functional.one_hot(torch.as_tensor(cl_label), num_classes=2).float()
                
                cls_loss = self.task_manager.cls_setting(
                    model, 
                    "execute",
                    inputs, 
                    cl_label
                )
                print(f"{epoch_index} - {step}/{len(train_loader)} - Loss: {cls_loss.item():.4f}")
                cls_train_loss += cls_loss.item()
            
            cls_train_loss /= step
            return cls_train_loss

        else:
            for batch_data in train_loader:
                step += 1
                inputs, seg_labels = batch_data["image"].to(device), batch_data["seg_label"].to(device)
                det_targets = [dict(
                                    box_label = batch_data["box_label"][0].to(device),
                                    box = batch_data["box"][0].to(device),
                                )
                        ]
                cl_label = batch_data["cls_label"][0].to(device)
                cl_label = torch.nn.functional.one_hot(torch.as_tensor(cl_label), num_classes=2).float()
                
                total_loss, seg_loss, cls_loss, det_loss = self.task_manager.multi_setting(
                    model, 
                    "execute",
                    self.multi_opt,
                    inputs, 
                    seg_labels,
                    cl_label,
                    det_targets,
                    step
                )
                
                gc.collect()
                torch.cuda.empty_cache()
                total_loss_all += total_loss.item()
                if isinstance(seg_loss, torch.Tensor):
                    seg_epoch_loss += seg_loss.item() 
                else:
                    seg_epoch_loss += seg_loss
                if isinstance(cls_loss, torch.Tensor):
                    cl_epoch_loss += cls_loss.item() 
                else:
                    cl_epoch_loss += cls_loss
                if isinstance(det_loss, torch.Tensor):
                    det_epoch_loss += det_loss.item() 
                else:
                    det_epoch_loss += det_loss
                print(f"{epoch_index} - {step}/{len(train_loader)} - Loss: {total_loss.item()}")
                     
            #Learning rate
            encoder_lr = self.task_manager.multi_lr_scheduler.get_lr()[0]
            seg_lr = self.task_manager.multi_lr_scheduler.get_lr()[1]
            det_lr = self.task_manager.multi_lr_scheduler.get_lr()[2]
            cls_lr = self.task_manager.multi_lr_scheduler.get_lr()[3]
            print(f"{epoch_index} - {step}/{len(train_loader)} - Enc Lr: {encoder_lr} - Seg Lr: {seg_lr} - Det Lr: {det_lr} - Cls Lr: {cls_lr}")
            print(f"Epoch {epoch_index} - Total Loss: {total_loss_all} - Seg Loss: {seg_epoch_loss} - Det Loss: {det_epoch_loss} - Cls Loss: {cl_epoch_loss}")

            self.task_manager.multi_lr_scheduler.step()
            total_loss_all /= step
            seg_epoch_loss /= step
            cl_epoch_loss /= step
            det_epoch_loss /= step
            return total_loss_all, seg_epoch_loss, cl_epoch_loss, det_epoch_loss
                

    def validate(self, model, task, epoch_index, validation_loader, device):
        model.eval()
        det_val_outputs_all = []
        det_val_targets_all = []
        val_epoch_loss = 0
        val_epoch_cls_loss = 0
        val_epoch_box_reg_loss = 0
        
        if task == "segmentation":
            with torch.no_grad():
                step = 0
                epoch_loss = 0
                threshold = 0.3
                for val_data in validation_loader:
                    step += 1
                    val_images, val_labels = val_data["image"].to(device), val_data["seg_label"].to(device)
                    
                    # No Sliding Window
                    seg_loss, seg_outputs = self.task_manager.seg_setting(
                        model, 
                        "execute",
                        val_images, 
                        val_labels
                    )
                    
                    epoch_loss += seg_loss.item()
                    val_outputs = (seg_outputs > threshold).float()
                    self.mean_dice_metric(y_pred=val_outputs, y=val_labels)
                    self.dice_metric(y_pred=val_outputs, y=val_labels)
                    self.hd95_metric(y_pred=val_outputs, y=val_labels)
                    self.hd95_metric_batch(y_pred=val_outputs, y=val_labels)
                
                metric_mean = self.mean_dice_metric.aggregate().item()
                metric_batch = self.dice_metric.aggregate()
                hd95_metric_mean = self.hd95_metric.aggregate().item()
                hd95_metric_batch = self.hd95_metric_batch.aggregate()
                
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
                self.mean_dice_metric.reset()
                self.dice_metric.reset()
                hd95_tc = hd95_metric_batch[0].item()
                hd95_wt = hd95_metric_batch[1].item()
                hd95_et = hd95_metric_batch[2].item()
                self.hd95_metric.reset()
                self.hd95_metric_batch.reset()

                print(
                        "Val {}/{} {}/{}".format(epoch_index, self.num_epochs, step, len(validation_loader)),
                        ", Dice_TC:",
                        metric_tc,
                        ", Dice_WT:",
                        metric_wt,
                        ", Dice_ET:",
                        metric_et,
                    ) 
                epoch_loss /= step
            return epoch_loss, metric_mean, metric_tc, metric_wt, metric_et, hd95_metric_mean, hd95_tc, hd95_wt, hd95_et
        
        elif task == "detection":
            with torch.no_grad():
                epoch_loss = 0
                step = 0
                for val_data in validation_loader:
                    step += 1
                    inputs = val_data["image"].to(device)
                    targets = [dict(
                                        box_label = val_data["box_label"][0].to(device),
                                        box = val_data["box"][0].to(device),
                                    )
                            ]

                    det_outputs, det_loss, det_losses = self.task_manager.det_setting(
                        model, 
                        "execute",
                        inputs, 
                        targets
                    )
                    
                    det_val_outputs_all += det_outputs
                    det_val_targets_all.append(targets[0])
                    val_epoch_loss += det_loss.detach().item()
                    val_epoch_cls_loss += det_losses[self.task_manager.cls_key].detach().item()
                    val_epoch_box_reg_loss += det_losses[self.task_manager.box_reg_key].detach().item()
                    print(f"{epoch_index} - {step}/{len(validation_loader)} - Loss: {det_loss.item():.4f}")
                    del inputs, targets, det_outputs, det_losses, det_loss

                val_epoch_loss /= step
                val_epoch_cls_loss /= step
                val_epoch_box_reg_loss /= step
            
            torch.cuda.empty_cache()
            results_metric = self.task_manager.matching_batch(
                iou_fn=self.task_manager.box_utils.box_iou,
                iou_thresholds=self.det_metric.iou_thresholds,
                pred_boxes=[
                    val_data_i[self.task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
                ],
                pred_classes=[
                    val_data_i[self.task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
                ],
                pred_scores=[
                    val_data_i[self.task_manager.pred_score_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
                ],
                gt_boxes=[val_data_i[self.task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_targets_all],
                gt_classes=[
                    val_data_i[self.task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_targets_all
                ],
            )
            val_epoch_metric_dict = self.det_metric(results_metric)[0]

            return val_epoch_loss, val_epoch_cls_loss, val_epoch_box_reg_loss, val_epoch_metric_dict
        
        elif task == "classification":
            cls_val_loss = 0
            val_acc = 0
            all_labels = []
            all_preds = []
            with torch.no_grad():
                step = 0
                for val_data in validation_loader:
                    step += 1
                    inputs = val_data["image"].to(device) 
                    cl_label = val_data["cls_label"][0].to(device)
                    cl_label_one_hot = torch.nn.functional.one_hot(torch.as_tensor(cl_label), num_classes=2).float()
                    
                    # No sliding windows
                    cls_loss, cls_outputs = self.task_manager.cls_setting(
                        model, 
                        "execute",
                        inputs, 
                        cl_label_one_hot
                    )

                    acc1 = accuracy(cls_outputs, cl_label, topk=(1,))
                    predicted_labels = torch.argmax(cls_outputs, dim=1)

                    inference_data_image = val_data["image"]
                    file_path = inference_data_image.meta["filename_or_obj"]
                    file_path = str(file_path)
                    cleaned_path = file_path.replace("[", "").replace("]", "").replace("'", "")
                    inference_img_filenames = cleaned_path.split("/")[-1]
                    file_name_parts = inference_img_filenames.split('.')
                    inference_img_filenames = file_name_parts[0]

                    cls_val_loss += cls_loss.item()
                    val_acc += acc1[0].item()
                    all_labels.extend(cl_label.cpu().numpy())
                    all_preds.extend(predicted_labels.cpu().numpy())
                    print(f"{epoch_index} - {step}/{len(validation_loader)} - Image name: {inference_img_filenames} - Loss: {cls_loss.item()} - Acc: {val_acc}")
                
                cls_val_loss /= step
                val_acc /= step 
                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)

                # Compute the overall metrics using sklearn functions
                acc, sensitivity, specificity = sensitivity_specificity(all_preds, all_labels) 
                print(f"{epoch_index} - Acc: {acc} - Sen: {sensitivity} - Spe: {specificity}")

            return cls_val_loss, acc, sensitivity, specificity
        
        else:
            with torch.no_grad():
                step = 0
                threshold = 0.3
                val_cl_acc = 0
                seg_epoch_loss = 0
                cl_epoch_loss = 0
                det_epoch_loss = 0 
                total_val_epoch_loss = 0   
                val_result ={}
                all_labels = []
                all_preds = []
                
                # 检查并初始化 task_manager（如果还没有初始化）
                if self.task_manager.seg_loss is None:
                    self.task_manager.multi_setting(model, "init", self.multi_opt)
                
                for val_data in validation_loader:
                    step += 1
                    val_images, seg_labels = val_data["image"].to(device), val_data["seg_label"].to(device)
                    det_targets = [dict(
                                        box_label = val_data["box_label"][0].to(device),
                                        box = val_data["box"][0].to(device),
                                    )
                            ]
                    cl_labels = val_data["cls_label"][0].to(device)
                    cl_labels_one_hot = torch.nn.functional.one_hot(torch.as_tensor(cl_labels), num_classes=2).float()

                    seg_loss, cls_loss, det_loss, seg_outputs, cls_outputs, det_outputs = self.task_manager.multi_setting(
                        model, 
                        "execute",
                        self.multi_opt,
                        val_images, 
                        seg_labels,
                        cl_labels_one_hot,
                        det_targets,
                        step
                    )
                    
                    total_val_loss = seg_loss + cls_loss + det_loss

                    #Seg and Det loss
                    total_val_epoch_loss += total_val_loss.item()
                    seg_epoch_loss += seg_loss.item()
                    cl_epoch_loss += cls_loss.item()
                    det_epoch_loss += det_loss.detach().item()

                    #Seg Metric
                    seg_outputs = (seg_outputs > threshold).float() # No sliding window
                    self.mean_dice_metric(y_pred=seg_outputs, y=seg_labels)
                    self.dice_metric(y_pred=seg_outputs, y=seg_labels)
                    self.hd95_metric(y_pred=seg_outputs, y=seg_labels)
                    self.hd95_metric_batch(y_pred=seg_outputs, y=seg_labels)

                    #Cls Metric
                    cl_acc1 = accuracy(cls_outputs, cl_labels, topk=(1,))
                    val_cl_acc += cl_acc1[0].item()
                    predicted_labels = torch.argmax(cls_outputs, dim=1)
                    all_labels.extend(cl_labels.cpu().numpy())
                    all_preds.extend(predicted_labels.cpu().numpy())
                    
                    #Det data
                    det_val_outputs_all += det_outputs
                    det_val_targets_all.append(det_targets[0])

                    print(f"{epoch_index} - {step}/{len(validation_loader)} - Loss: {total_val_loss.item():.4f}")

                #Epoch loss
                total_val_epoch_loss /= step
                seg_epoch_loss /= step
                cl_epoch_loss /= step
                det_epoch_loss /= step

                #Epoch seg metric
                metric_mean = self.mean_dice_metric.aggregate().item()
                metric_batch = self.dice_metric.aggregate()
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
                self.mean_dice_metric.reset()
                self.dice_metric.reset()

                hd95_metric_mean = self.hd95_metric.aggregate().item()
                hd95_metric_batch = self.hd95_metric_batch.aggregate()
                hd95_tc = hd95_metric_batch[0].item()
                hd95_wt = hd95_metric_batch[1].item()
                hd95_et = hd95_metric_batch[2].item()
                self.hd95_metric.reset()
                self.hd95_metric_batch.reset()
                
                #Epoch cls Metric
                val_cl_acc /= step 
                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)
                acc, sensitivity, specificity = sensitivity_specificity(all_preds, all_labels) 
                print(f"{epoch_index} - Acc: {acc} - Sen: {sensitivity} - Spe: {specificity}")

                #Epoch det Metric
                torch.cuda.empty_cache()
                results_metric = self.task_manager.matching_batch(
                    iou_fn=self.task_manager.box_utils.box_iou,
                    iou_thresholds=self.det_metric.iou_thresholds,
                    pred_boxes=[
                        val_data_i[self.task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
                    ],
                    pred_classes=[
                        val_data_i[self.task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
                    ],
                    pred_scores=[
                        val_data_i[self.task_manager.pred_score_key].cpu().detach().numpy() for val_data_i in det_val_outputs_all
                    ],
                    gt_boxes=[val_data_i[self.task_manager.target_box_key].cpu().detach().numpy() for val_data_i in det_val_targets_all],
                    gt_classes=[
                        val_data_i[self.task_manager.target_label_key].cpu().detach().numpy() for val_data_i in det_val_targets_all
                    ],
                )
                val_det_epoch_metric_dict = self.det_metric(results_metric)[0]
                val_result['total_val_epoch_loss'] = total_val_epoch_loss
                val_result['seg_epoch_loss'] = seg_epoch_loss
                val_result['cl_epoch_loss'] = cl_epoch_loss
                val_result['det_epoch_loss'] = det_epoch_loss
                val_result['metric_mean'] = metric_mean
                val_result['metric_tc'] = metric_tc
                val_result['metric_wt'] = metric_wt
                val_result['metric_et'] = metric_et
                val_result['hd_mean'] = hd95_metric_mean
                val_result['hd_tc'] = hd95_tc
                val_result['hd_wt'] = hd95_wt
                val_result['hd_et'] = hd95_et
                val_result['cl_acc'] = acc
                val_result['sensitivity'] = sensitivity
                val_result['specificity'] = specificity
                val_result['val_det_epoch_metric_dict'] = val_det_epoch_metric_dict
                det_metric_list = [(key, round(value, 4)) for key, value in val_det_epoch_metric_dict.items()]

                return val_result, det_metric_list


    def swin_train(self):
        num_epochs=self.num_epochs
        best_val_metric=0
                
        if self.task == "segmentation": 
            train_loss_array=[]
            val_loss_array=[]
            val_metric_array=[]
            metric_values_tc = []
            metric_values_wt = []
            metric_values_et = []
            hd95_mean = []
            hd95_value_tc = []
            hd95_value_wt = []
            hd95_value_et = []
            f = open(os.path.join(self.root_dir, 'epoch_result.txt'), 'a+')
            g = open(os.path.join(self.root_dir, 'epoch_semantic_class.txt'), 'a+')
            print(f"Traing set: {len(self.train_loader.dataset.data)}; Validation set: {len(self.valid_loader.dataset.data)}")

            for epoch in range(num_epochs):
                train_loss = self.train_one_epoch(self.model, self.task, epoch, self.train_loader, self.device)
                val_loss, val_metric, metric_tc, metric_wt, metric_et, hd95_metric_mean, hd95_tc, hd95_wt, hd95_et = self.validate(self.model, self.task, epoch, self.valid_loader, self.device)
                        
                train_loss_array.append(train_loss)
                val_loss_array.append(val_loss)
                val_metric_array.append(val_metric)
                metric_values_tc.append(metric_tc)
                metric_values_wt.append(metric_wt)
                metric_values_et.append(metric_et)
                hd95_mean.append(hd95_metric_mean)
                hd95_value_tc.append(hd95_tc)
                hd95_value_wt.append(hd95_wt)
                hd95_value_et.append(hd95_et)

                print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Metric: {np.mean(val_metric)}") 
                f.write('Epoch {%d}, Train Loss: {%.4f}, Val Loss: {%.4f}, Val Metric: {%.4f}\n' % (epoch+1, train_loss, val_loss, val_metric))
                g.write(f"Epoch {epoch+1}, Mean_Dice: {val_metric}, Dice_Val_TC: {metric_tc}, Dice_Val_WT: {metric_wt}, Dice_Val_ET: {metric_et}, HD_mean: {hd95_metric_mean}, HD_TC: {hd95_tc}, HD_WT: {hd95_wt}, HD_ET: {hd95_et}\n")

                # if val_metric > best_val_metric and val_metric > 0.75:
                if val_metric > best_val_metric and val_metric > 0.70:
                    best_val_metric = val_metric
                    best_net = self.model.state_dict()
                    self.segmentation_path = os.path.join(self.root_dir, '%d-%.4f-seg.pkl' % (
                        epoch+1, val_metric))
                    torch.save(best_net, self.segmentation_path)
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}\n".format(best_val_metric, val_metric)
                    )
            
            # Loss Curve 
            x = range(num_epochs)
            self.plot_multiple_curves(
                x=x,
                y_list=[train_loss_array, val_loss_array],
                labels=["Train Loss", "Validation Loss"],
                xlabel="Epoch",
                ylabel="Loss",
                title="Train and Validation Loss Curve",
                colors=["blue", "orange"],
                save_path=os.path.join(self.root_dir, "loss_curves.png")
            )

            # Dice Curves
            self.plot_subplots(
                x=x,
                y_list=[metric_values_tc, metric_values_wt, metric_values_et],
                titles=["Val Mean Dice TC", "Val Mean Dice WT", "Val Mean Dice ET"],
                xlabel="epoch",
                ylabel="Dice Score",
                colors=["blue", "orange", "purple"],
                save_path=os.path.join(self.root_dir, "tumor_dice.png")
            )

            # HD Curves
            self.plot_subplots(
                x=x,
                y_list=[hd95_value_tc, hd95_value_wt, hd95_value_et],
                titles=["Val HD TC", "Val HD WT", "Val HD ET"],
                xlabel="epoch",
                ylabel="HD95 Value",
                colors=["blue", "orange", "purple"],
                save_path=os.path.join(self.root_dir, "tumor_hd.png")
            )

        elif self.task == "detection":
            val_mAP_IoU = []
            val_mAR_IoU = []
            train_det_loss = []
            val_det_loss = []
            train_det_cls_loss = []
            train_det_reg_loss = []
            val_det_cls_loss = []
            val_det_reg_loss = []
            det_val_metric = []
            h = open(os.path.join(self.root_dir, 'det_train_epoch.txt'), 'a+')
            k = open(os.path.join(self.root_dir, 'det_validation_epoch.txt'), 'a+')
            m = open(os.path.join(self.root_dir, 'det_metric.txt'), 'a+')
            print(f"Traing set: {len(self.train_loader.dataset.data)}; Validation set: {len(self.valid_loader.dataset.data)}")

            for epoch in range(num_epochs):
                det_epoch_loss, epoch_cls_loss, epoch_box_reg_loss = self.train_one_epoch(self.model, self.task, epoch, self.train_loader, self.device)
                val_epoch_loss, val_epoch_cls_loss, val_epoch_box_reg_loss, val_epoch_metric_dict = self.validate(self.model, self.task, epoch, self.valid_loader, self.device)

                val_map = val_epoch_metric_dict['mAP_IoU_0.10_0.50_0.05_MaxDet_10']
                val_mar = val_epoch_metric_dict['mAR_IoU_0.10_0.50_0.05_MaxDet_10']
                val_epoch_metric = val_epoch_metric_dict.values()
                val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
                
                val_mAP_IoU.append(val_map)
                val_mAR_IoU.append(val_mar)
                train_det_loss.append(det_epoch_loss)
                train_det_cls_loss.append(epoch_cls_loss)
                train_det_reg_loss.append(epoch_box_reg_loss)
                val_det_loss.append(val_epoch_loss)
                val_det_cls_loss.append(val_epoch_cls_loss)
                val_det_reg_loss.append(val_epoch_box_reg_loss)
                det_val_metric.append(val_epoch_metric)
                
                print(f"Epoch {epoch+1}, avg_train_loss: {det_epoch_loss}, avg_train_cls_loss: {epoch_cls_loss}, avg_train_box_reg_loss: {epoch_box_reg_loss}")
                print(f"Epoch {epoch+1}, avg_val_loss: {val_epoch_loss}, avg_val_cls_loss:{val_epoch_cls_loss}, avg_val_box_reg_loss: {val_epoch_box_reg_loss}, avg_val_metric: {val_epoch_metric}, mAP_IoU: {val_map}, mAR_IoU: {val_mar}")
                h.write(f"Epoch {epoch+1}, avg_train_loss: {det_epoch_loss}, avg_train_cls_loss: {epoch_cls_loss}, avg_train_box_reg_loss: {epoch_box_reg_loss}\n")
                k.write(f"Epoch {epoch+1}, avg_val_loss: {val_epoch_loss}, avg_val_cls_loss:{val_epoch_cls_loss}, avg_val_box_reg_loss: {val_epoch_box_reg_loss}, avg_val_metric: {val_epoch_metric}\n")
                m.write(f"Epoch {epoch+1}, mAP_IoU: {val_map}, mAR_IoU: {val_mar}\n")

                # save best trained model
                if val_epoch_metric > best_val_metric:
                    best_val_metric = val_epoch_metric
                    best_net = self.model.state_dict()
                    self.detection_path = os.path.join(self.root_dir, '%d-%.4f-%.4f.pkl' % (
                        epoch+1, val_epoch_metric, val_map))
                    torch.save(best_net, self.detection_path)
                    print(
                        "Model Was Saved ! Current Best Avg. Metric: {} Current Avg. Metric:{}\n".format(best_val_metric, val_epoch_metric)
                    )
              
            # Loss Curve   
            x = range(num_epochs)   
            self.plot_multiple_curves(
                x=x,
                y_list=[train_det_loss, val_det_loss],
                labels=["Train Loss", "Validation Loss"],
                xlabel="Epoch",
                ylabel="Loss",
                title="Training and Validation Loss Curve",
                colors=["blue", "orange"],
                save_path=os.path.join(self.root_dir, "loss_curves.png")
            )

            # Metric Curve
            self.plot_multiple_curves(
                x=x,
                y_list=[val_mAP_IoU, val_mAR_IoU],
                labels=["mAP", "mAR"],
                xlabel="Epoch",
                ylabel="Metric",
                title="Validation mAP and mAR Curve",
                colors=["blue", "orange"],
                save_path=os.path.join(self.root_dir, "metric_curves.png")
            )
        
        elif self.task == "classification":
            train_cls_loss = []
            val_cls_loss = []
            val_cls_acc1 = []
            val_cls_sen = []
            val_cls_spe = []
            best_cls_acc = 0

            c = open(os.path.join(self.root_dir, 'cls_loss_acc.txt'), 'a+')

            for epoch in range(num_epochs):
                cls_train_loss = self.train_one_epoch(self.model, self.task, epoch, self.train_loader, self.device)
                cls_val_loss, val_acc, sensitivity, specificity = self.validate(self.model, self.task, epoch, self.valid_loader, self.device)
                
                train_cls_loss.append(cls_train_loss)
                val_cls_loss.append(cls_val_loss)
                val_cls_acc1.append(val_acc)
                val_cls_sen.append(sensitivity)
                val_cls_spe.append(specificity)

                c.write(f"Epoch {epoch+1}, training loss: {cls_train_loss}, validation loss: {cls_val_loss}, accuracy: {val_acc}\n")
                if val_acc > best_cls_acc:
                    best_cls_acc = val_acc
                    best_net = self.model.state_dict()
                    self.cls_path = os.path.join(self.root_dir, '%d-%.4f-cls.pkl' % (
                        epoch, val_acc))
                    torch.save(best_net, self.cls_path)
                    print(
                        "Model Was Saved ! Current Best Avg. Metric: {} Current Avg. Metric:{}\n".format(best_cls_acc, val_acc)
                    )
                    
            x = range(num_epochs)  
            # Loss Curves 
            self.plot_multiple_curves(
                x=x,
                y_list=[train_cls_loss, val_cls_loss],
                labels=["Train Loss", "Validation Loss"],
                xlabel="Epoch",
                ylabel="Loss",
                title="Training and Validation Loss Curve",
                colors=["blue", "orange"],
                save_path=os.path.join(self.root_dir, "loss_curve.png")
            )

            # Accuracy, Sensitivity and Specificity Curves
            self.plot_subplots(
                x=x,
                y_list=[val_cls_acc1, val_cls_sen, val_cls_spe],
                titles=["Validation Accuracy", "Validation Sensitivity", "Validation Specificity"],
                xlabel="Epoch",
                ylabel="Metric",
                colors=["blue", "orange", "green"],
                save_path=os.path.join(self.root_dir, "cls_metrics_curve.png")
            )
        
        else:
            train_total_loss = []
            train_seg_loss = []
            train_cl_loss = []
            train_det_loss =[]
            train_det_cls_loss = []
            train_det_reg_loss = []

            val_total_loss = []
            val_seg_loss = []
            val_cl_loss = []
            val_det_loss =[]
            val_det_cls_loss = []
            val_det_reg_loss = []
            
            seg_epoch_metric = []
            metric_values_tc = []
            metric_values_wt = []
            metric_values_et = []
            seg_hd_mean = []
            seg_hd_tc = []
            seg_hd_wt = []
            seg_hd_et = []
            cl_acc = []
            cl_sen = []
            cl_spe = []
            det_val_metric = []
            val_mAP_IoU = []
            val_mAR_IoU = []

            h = open(os.path.join(self.root_dir, 'epoch_loss.txt'), 'a+')
            m = open(os.path.join(self.root_dir, 'val_metric.txt'), 'a+')
            n = open(os.path.join(self.root_dir, 'det_metric.txt'), 'a+')
            print(f"Traing set: {len(self.train_loader.dataset.data)}; Validation set: {len(self.valid_loader.dataset.data)}")
            
            self.task_manager.multi_setting(self.model, "init", self.multi_opt)
            for epoch in range(num_epochs):
                total_loss, seg_epoch_loss, cl_epoch_loss, det_epoch_loss = self.train_one_epoch(self.model, self.task, epoch, self.train_loader, self.device)
                # total_loss, seg_epoch_loss, det_epoch_loss, cl_epoch_loss = 0,0,0,0
                val_epoch_results, det_metric_list = self.validate(self.model, self.task, epoch, self.valid_loader, self.device)
                # det_epoch_loss, epoch_cls_loss, epoch_box_reg_loss = 0, 0, 0
                
                train_total_loss.append(total_loss)
                train_seg_loss.append(seg_epoch_loss)
                train_cl_loss.append(cl_epoch_loss)
                train_det_loss.append(det_epoch_loss)

                val_total_loss.append(val_epoch_results['total_val_epoch_loss'])
                val_seg_loss.append(val_epoch_results['seg_epoch_loss'])
                val_cl_loss.append(val_epoch_results['cl_epoch_loss'])
                val_det_loss.append(val_epoch_results['det_epoch_loss'])

                seg_epoch_metric.append(val_epoch_results['metric_mean'])
                metric_values_tc.append(val_epoch_results['metric_tc'])
                metric_values_wt.append(val_epoch_results['metric_wt'])
                metric_values_et.append(val_epoch_results['metric_et'])
                seg_hd_mean.append(val_epoch_results['hd_mean'])
                seg_hd_tc.append(val_epoch_results['hd_tc'])
                seg_hd_wt.append(val_epoch_results['hd_wt'])
                seg_hd_et.append(val_epoch_results['hd_et'])
                cl_acc.append(val_epoch_results['cl_acc'])
                cl_sen.append(val_epoch_results['sensitivity'])
                cl_spe.append(val_epoch_results['specificity'])

                val_mAP_IoU.append(val_epoch_results['val_det_epoch_metric_dict']['mAP_IoU_0.10_0.50_0.05_MaxDet_10'])
                val_mAR_IoU.append(val_epoch_results['val_det_epoch_metric_dict']['mAR_IoU_0.10_0.50_0.05_MaxDet_10'])
                det_val_epoch_metric = val_epoch_results['val_det_epoch_metric_dict'].values()
                det_val_epoch_metric = sum(det_val_epoch_metric) / len(det_val_epoch_metric)
                det_val_metric.append(det_val_epoch_metric)

                val_epoch_metric = (det_val_metric[epoch] + seg_epoch_metric[epoch] + cl_acc[epoch])/3
                # if val_epoch_metric > best_val_metric and val_epoch_metric > 0.75:
                if val_epoch_metric > best_val_metric and val_epoch_metric > 0.6:
                    best_val_metric = val_epoch_metric
                    best_net = self.model.state_dict()
                    self.multiswin_path = os.path.join(self.root_dir, '%d-s%.4f-d%.4f-c%.4f-%.4f.pkl' % (
                        epoch+1, seg_epoch_metric[epoch], det_val_metric[epoch], cl_acc[epoch], val_epoch_metric))
                    torch.save(best_net, self.multiswin_path)
                    print(
                        "Model Was Saved ! Current Best Avg. Metric: {} Current Avg. Metric:{}\n".format(best_val_metric, val_epoch_metric)
                    )

                print(f"Epoch {epoch+1}, Train Total Loss: {total_loss}, Val Loss: {val_epoch_results['total_val_epoch_loss']}, Dice: {np.mean(val_epoch_results['metric_mean'])}, Acc: {val_epoch_results['cl_acc']}, Det Metric: {det_val_epoch_metric}\n")
                h.write(f"Epoch {epoch+1}, avg_train_loss: {total_loss}, seg_train_loss: {seg_epoch_loss}, cl_train_loss: {cl_epoch_loss}, det_train_loss: {det_epoch_loss}\n")
                h.write(f"Epoch {epoch+1}, avg_val_loss: {val_epoch_results['total_val_epoch_loss']}, seg_val_loss: {val_epoch_results['seg_epoch_loss']}, cl_val_loss: {val_epoch_results['cl_epoch_loss']}, det_val_loss: {val_epoch_results['det_epoch_loss']}\n")
                m.write(f"Epoch {epoch+1}, Segmentation avg_mean_dice: {val_epoch_results['metric_mean']},  Dice_Val_WT: {val_epoch_results['metric_wt']}; Dice_Val_TC: {val_epoch_results['metric_tc']}, Dice_Val_EH: {val_epoch_results['metric_et']}\n")
                m.write(f"Epoch {epoch+1}, Segmentation mean_hd: {val_epoch_results['hd_mean']},  hd_wt: {val_epoch_results['hd_wt']}; hd_tc: {val_epoch_results['hd_tc']}, hd_et: {val_epoch_results['hd_et']}\n")
                m.write(f"Epoch {epoch+1}, Clssification accuracy: {val_epoch_results['cl_acc']}, Sen: {val_epoch_results['sensitivity']}, Spe: {val_epoch_results['specificity']}\n")
                m.write(f"Epoch {epoch+1}, Detection avg_det_metric: {det_val_metric[epoch]}, val_mAP_IoU: {val_epoch_results['val_det_epoch_metric_dict']['mAP_IoU_0.10_0.50_0.05_MaxDet_10']}, val_mAR_IoU: {val_epoch_results['val_det_epoch_metric_dict']['mAR_IoU_0.10_0.50_0.05_MaxDet_10']}\n")

                line = f"Epoch {epoch+1}, "
                line += ", ".join([f"{key}: {value}" for key, value in val_epoch_results['val_det_epoch_metric_dict'].items()])
                n.write(line + "\n")
                
            # Training Loss
            self.plot_multiple_curves(
                x=list(range(len(train_total_loss))),
                y_list=[train_total_loss, train_seg_loss, train_cl_loss, train_det_loss],
                labels=["Total Loss", "Segmentation Loss", "Classification Loss", "Detection Loss"],
                xlabel="Epochs",
                ylabel="Loss",
                title="Loss Curves",
                colors=["blue", "orange", "green", "red"],
                save_path=os.path.join(self.root_dir, "train_loss_curves.png")
            )
            # Validation Loss
            self.plot_multiple_curves(
                x=list(range(len(val_total_loss))),
                y_list=[val_total_loss, val_seg_loss, val_cl_loss, val_det_loss],
                labels=["Total Loss", "Segmentation Loss", "Classification Loss", "Detection Loss"],
                xlabel="Epochs",
                ylabel="Loss",
                title="Loss Curves",
                colors=["blue", "orange", "green", "red"],
                save_path=os.path.join(self.root_dir, "val_loss_curves.png")
            )
            # Dice 
            self.plot_multiple_curves(
                x=list(range(len(seg_epoch_metric))),
                y_list=[seg_epoch_metric, metric_values_tc, metric_values_wt, metric_values_et],
                labels=["Dice Mean", "Dice TC", "Dice WT", "Dice ET"],
                xlabel="Epochs",
                ylabel="Dice Score",
                title="Dice Metric Curves",
                colors=["blue", "orange", "green", "red"],
                save_path=os.path.join(self.root_dir, "dice_metrics.png")
            )
            # HD
            self.plot_multiple_curves(
                x=list(range(len(seg_hd_mean))),
                y_list=[seg_hd_mean, seg_hd_tc, seg_hd_wt, seg_hd_et],
                labels=["HD Mean", "HD TC", "HD WT", "HD ET"],
                xlabel="Epochs",
                ylabel="Hausdorff Distance",
                title="Hausdorff Distance Curves",
                colors=["blue", "orange", "green", "red"],
                save_path=os.path.join(self.root_dir, "hd_metrics.png")
            )
            # Detection Metrics
            self.plot_multiple_curves(
                x=list(range(len(val_mAP_IoU))),
                y_list=[val_mAP_IoU, val_mAR_IoU],
                labels=["mAP", "mAR"],
                xlabel="Epochs",
                ylabel="Metric Value",
                title="Detection Metrics (mAP and mAR)",
                colors=["blue", "orange"],
                save_path=os.path.join(self.root_dir, "det_metrics.png")
            )
            # Classification Metircs
            self.plot_multiple_curves(
                x=list(range(len(cl_acc))),
                y_list=[cl_acc, cl_sen, cl_spe],
                labels=["Accuracy", "Sensitivity", "Specificity"],
                xlabel="Epochs",
                ylabel="Metric Value",
                title="Classification Metrics (Acc, Sen, Spe)",
                colors=["blue", "orange", "green"],
                save_path=os.path.join(self.root_dir, "cls_metrics.png")
            )