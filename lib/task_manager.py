from __future__ import annotations
import warnings
from collections.abc import Callable, Sequence
from typing import Any
import torch
from torch import Tensor, nn
from collections import OrderedDict
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from monai.apps.detection.utils.ATSS_matcher import ATSSMatcher
from monai.apps.detection.utils.box_coder import BoxCoder
from monai.apps.detection.utils.box_selector import BoxSelector
from monai.apps.detection.utils.detector_utils import check_training_targets, preprocess_images
from monai.apps.detection.utils.hard_negative_sampler import HardNegativeSampler
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_, predict_with_inferer
from monai.data.box_utils import box_iou
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import resnet
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple_rep, optional_import
from monai.losses import DiceLoss, FocalLoss
from torch.autograd import Variable
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils, decollate_batch
BalancedPositiveNegativeSampler, _ = optional_import(
    "torchvision.models.detection._utils", name="BalancedPositiveNegativeSampler"
)
Matcher, _ = optional_import("torchvision.models.detection._utils", name="Matcher")

class TaskManager:
    def __init__(self, task_mode="multi-task", device="cuda", num_epoch = 100):
        """
        Initialize TaskManager.
        
        Args:
            task_mode (str): "single-task" for a single task or "multi-task" for all tasks.
            device (str): Device to run on.
        """
        self.task_mode = task_mode 
        self.device = device
        self.num_epoch = num_epoch
        self.optimizer2 = None
        self.weights = None
        self.l0 = None
        self.T = None

        # Components
        self.seg_optimizer = None
        self.seg_lr_scheduler = None
        self.seg_loss = None
        
        self.cls_optimizer = None
        self.cls_lr_scheduler = None
        self.cls_loss = None
    
        self.det_optimizer = None
        self.det_lr_scheduler = None
        self.det_loss = None

        self.multi_optimizer = None
        self.multi_lr_scheduler = None

    def get_attribute_from_network(self, model, attr_name, default_value=None):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
        elif default_value is not None:
            return default_value
        else:
            raise ValueError(f"network does not have attribute {attr_name}, please provide it in the detector.")
    
    def generate_anchors(self, images: Tensor, head_outputs: dict[str, list[Tensor]]) -> None:
        """
        Generate anchors and store it in self.anchors: List[Tensor].
        We generate anchors only when there is no stored anchors,
        or the new coming images has different shape with self.previous_image_shape

        Args:
            images: input images, a (B, C, H, W) or (B, C, H, W, D) Tensor.
            head_outputs: head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
            sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
            sized (B, sum(HW(D)A), 2*self.spatial_dims)
        """
        if (self.anchors is None) or (self.previous_image_shape != images.shape):
            self.anchors = self.anchor_generator(images, head_outputs[self.cls_key])  # List[Tensor], len = batchsize
            self.previous_image_shape = images.shape
        
    def _reshape_maps(self, result_maps: list[Tensor]) -> Tensor:
        """
        Concat network output map list to a single Tensor.
        This function is used in both training and inference.

        Args:
            result_maps: a list of Tensor, each Tensor is a (B, num_channel*A, H, W) or (B, num_channel*A, H, W, D) map.
                A = self.num_anchors_per_loc

        Return:
            reshaped and concatenated result, sized (B, sum(HWA), num_channel) or (B, sum(HWDA), num_channel)
        """
        all_reshaped_result_map = []

        for result_map in result_maps:
            batch_size = result_map.shape[0]
            num_channel = result_map.shape[1] // self.num_anchors_per_loc
            spatial_size = result_map.shape[-self.spatial_dims :]

            # reshaped_result_map will become (B, A, num_channel, H, W) or (B, A, num_channel, H, W, D)
            # A = self.num_anchors_per_loc
            view_shape = (batch_size, -1, num_channel) + spatial_size
            reshaped_result_map = result_map.view(view_shape)

            # permute output to (B, H, W, A, num_channel) or (B, H, W, D, A, num_channel)
            if self.spatial_dims == 2:
                reshaped_result_map = reshaped_result_map.permute(0, 3, 4, 1, 2)
            elif self.spatial_dims == 3:
                reshaped_result_map = reshaped_result_map.permute(0, 3, 4, 5, 1, 2)
            else:
                ValueError("Images can only be 2D or 3D.")

            # reshaped_result_map will become (B, HWA, num_channel) or (B, HWDA, num_channel)
            reshaped_result_map = reshaped_result_map.reshape(batch_size, -1, num_channel)

            if torch.isnan(reshaped_result_map).any() or torch.isinf(reshaped_result_map).any():
                if torch.is_grad_enabled():
                    raise ValueError("Concatenated result is NaN or Inf.")
                else:
                    warnings.warn("Concatenated result is NaN or Inf.")

            all_reshaped_result_map.append(reshaped_result_map)

        return torch.cat(all_reshaped_result_map, dim=1)
    
    def postprocess_detections(
        self,
        head_outputs_reshape: dict[str, Tensor],
        anchors: list[Tensor],
        image_sizes: list[list[int]],
        num_anchor_locs_per_level: Sequence[int],
        need_sigmoid: bool = True,
    ) -> list[dict[str, Tensor]]:
        """
        Postprocessing to generate detection result from classification logits and box regression.
        Use self.box_selector to select the final output boxes for each image.

        Args:
            head_outputs_reshape: reshaped head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
            sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
            sized (B, sum(HW(D)A), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.

        Return:
            a list of dict, each dict corresponds to detection result on image.
        """

        # recover level sizes, HWA or HWDA for each level
        num_anchors_per_level = [
            num_anchor_locs * self.num_anchors_per_loc for num_anchor_locs in num_anchor_locs_per_level
        ]

        # split outputs per level
        split_head_outputs: dict[str, list[Tensor]] = {}
        for k in head_outputs_reshape:
            split_head_outputs[k] = list(head_outputs_reshape[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]  # List[List[Tensor]]

        class_logits = split_head_outputs[self.cls_key]  # List[Tensor], each sized (B, HWA, self.num_classes)
        box_regression = split_head_outputs[self.box_reg_key]  # List[Tensor], each sized (B, HWA, 2*spatial_dims)
        compute_dtype = class_logits[0].dtype

        num_images = len(image_sizes)  # B

        detections: list[dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [
                br[index] for br in box_regression
            ]  # List[Tensor], each sized (HWA, 2*spatial_dims)
            logits_per_image = [cl[index] for cl in class_logits]  # List[Tensor], each sized (HWA, self.num_classes)
            anchors_per_image, img_spatial_size = split_anchors[index], image_sizes[index]
            # decode box regression into boxes
            boxes_per_image = [
                self.box_coder.decode_single(b.to(torch.float32), a).to(compute_dtype)
                for b, a in zip(box_regression_per_image, anchors_per_image)
            ]  # List[Tensor], each sized (HWA, 2*spatial_dims)

            selected_boxes, selected_scores, selected_labels = self.box_selector.select_boxes_per_image(
                boxes_per_image, logits_per_image, img_spatial_size
            )

            detections.append(
                {
                    self.target_box_key: selected_boxes,  # Tensor, sized (N, 2*spatial_dims)
                    self.pred_score_key: selected_scores,  # Tensor, sized (N, )
                    self.target_label_key: selected_labels,  # Tensor, sized (N, )
                }
            )

        return detections

    def set_atss_matcher(self, num_candidates: int = 4, center_in_gt: bool = False) -> None:
        """
        Using for training. Set ATSS matcher that matches anchors with ground truth boxes

        Args:
            num_candidates: number of positions to select candidates from.
                Smaller value will result in a higher matcher threshold and less matched candidates.
            center_in_gt: If False (default), matched anchor center points do not need
                to lie withing the ground truth box. Recommend False for small objects.
                If True, will result in a strict matcher and less matched candidates.
        """
        self.proposal_matcher = ATSSMatcher(num_candidates, self.box_overlap_metric, center_in_gt, debug=False)
        
    def set_box_regression_loss(self, box_loss: nn.Module, encode_gt: bool, decode_pred: bool) -> None:
        """
        Using for training. Set loss for box regression.

        Args:
            box_loss: loss module for box regression
            encode_gt: if True, will encode ground truth boxes to target box regression
                before computing the losses. Should be True for L1 loss and False for GIoU loss.
            decode_pred: if True, will decode predicted box regression into predicted boxes
                before computing losses. Should be False for L1 loss and True for GIoU loss.

        Example:
            .. code-block:: python

                detector.set_box_regression_loss(
                    torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"),
                    encode_gt = True, decode_pred = False
                )
                detector.set_box_regression_loss(
                    monai.losses.giou_loss.BoxGIoULoss(reduction="mean"),
                    encode_gt = False, decode_pred = True
                )
        """
        self.box_loss_func = box_loss
        self.encode_gt = encode_gt
        self.decode_pred = decode_pred
    
    def compute_loss(
        self,
        head_outputs_reshape: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        anchors: list[Tensor],
        num_anchor_locs_per_level: Sequence[int],
    ) -> dict[str, Tensor]:
        """
        Compute losses.

        Args:
            head_outputs_reshape: reshaped head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.

        Return:
            a dict of several kinds of losses.
        """
        matched_idxs = self.compute_anchor_matched_idxs(anchors, targets, num_anchor_locs_per_level)
        losses_cls = self.compute_cls_loss(head_outputs_reshape[self.cls_key], targets, matched_idxs)
        losses_box_regression = self.compute_box_loss(
            head_outputs_reshape[self.box_reg_key], targets, anchors, matched_idxs
        )
        return {self.cls_key: losses_cls, self.box_reg_key: losses_box_regression}
    
    def compute_anchor_matched_idxs(
        self, anchors: list[Tensor], targets: list[dict[str, Tensor]], num_anchor_locs_per_level: Sequence[int]
    ) -> list[Tensor]:
        """
        Compute the matched indices between anchors and ground truth (gt) boxes in targets.
        output[k][i] represents the matched gt index for anchor[i] in image k.
        Suppose there are M gt boxes for image k. The range of it output[k][i] value is [-2, -1, 0, ..., M-1].
        [0, M - 1] indicates this anchor is matched with a gt box,
        while a negative value indicating that it is not matched.

        Args:
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            num_anchor_locs_per_level: each element represents HW or HWD at this level.


        Return:
            a list of matched index `matched_idxs_per_image` (Tensor[int64]), Tensor sized (sum(HWA),) or (sum(HWDA),).
            Suppose there are M gt boxes. `matched_idxs_per_image[i]` is a matched gt index in [0, M - 1]
            or a negative value indicating that anchor i could not be matched.
            BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
        """
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # anchors_per_image: Tensor, targets_per_image: Dice[str, Tensor]
            if targets_per_image[self.target_box_key].numel() == 0:
                # if no GT boxes
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            # matched_idxs_per_image (Tensor[int64]): Tensor sized (sum(HWA),) or (sum(HWDA),)
            # Suppose there are M gt boxes. matched_idxs_per_image[i] is a matched gt index in [0, M - 1]
            # or a negative value indicating that anchor i could not be matched.
            # BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
            if isinstance(self.proposal_matcher, Matcher):
                # if torchvision matcher
                match_quality_matrix = self.box_overlap_metric(
                    targets_per_image[self.target_box_key].to(anchors_per_image.device), anchors_per_image
                )
                matched_idxs_per_image = self.proposal_matcher(match_quality_matrix)
            elif isinstance(self.proposal_matcher, ATSSMatcher):
                # if monai ATSS matcher
                match_quality_matrix, matched_idxs_per_image = self.proposal_matcher(
                    targets_per_image[self.target_box_key].to(anchors_per_image.device),
                    anchors_per_image,
                    num_anchor_locs_per_level,
                    self.num_anchors_per_loc,
                )
            else:
                raise NotImplementedError(
                    "Currently support torchvision Matcher and monai ATSS matcher. Other types of matcher not supported. "
                    "Please override self.compute_anchor_matched_idxs(*) for your own matcher."
                )

            # if self.debug:
            #     print(f"Max box overlap between anchors and gt boxes: {torch.max(match_quality_matrix,dim=1)[0]}.")

            if torch.max(matched_idxs_per_image) < 0:
                warnings.warn(
                    f"No anchor is matched with GT boxes. Please adjust matcher setting, anchor setting,"
                    " or the network setting to change zoom scale between network output and input images."
                    f"GT boxes are {targets_per_image[self.target_box_key]}."
                )

            matched_idxs.append(matched_idxs_per_image)
        return matched_idxs

    def compute_cls_loss(
        self, cls_logits: Tensor, targets: list[dict[str, Tensor]], matched_idxs: list[Tensor]
    ) -> Tensor:
        """
        Compute classification losses.

        Args:
            cls_logits: classification logits, sized (B, sum(HW(D)A), self.num_classes)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            matched_idxs: a list of matched index. each element is sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            classification losses.
        """
        total_cls_logits_list = []
        total_gt_classes_target_list = []
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # for each image, get training samples
            sampled_cls_logits_per_image, sampled_gt_classes_target = self.get_cls_train_sample_per_image(
                cls_logits_per_image, targets_per_image, matched_idxs_per_image
            )
            total_cls_logits_list.append(sampled_cls_logits_per_image)
            total_gt_classes_target_list.append(sampled_gt_classes_target)

        total_cls_logits = torch.cat(total_cls_logits_list, dim=0)
        total_gt_classes_target = torch.cat(total_gt_classes_target_list, dim=0)
        losses: Tensor = self.cls_loss_func(total_cls_logits, total_gt_classes_target).to(total_cls_logits.dtype)
        return losses

    def compute_box_loss(
        self,
        box_regression: Tensor,
        targets: list[dict[str, Tensor]],
        anchors: list[Tensor],
        matched_idxs: list[Tensor],
    ) -> Tensor:
        """
        Compute box regression losses.

        Args:
            box_regression: box regression results, sized (B, sum(HWA), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            matched_idxs: a list of matched index. each element is sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            box regression losses.
        """
        total_box_regression_list = []
        total_target_regression_list = []

        for targets_per_image, box_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, box_regression, anchors, matched_idxs
        ):
            # for each image, get training samples
            decode_box_regression_per_image, matched_gt_boxes_per_image = self.get_box_train_sample_per_image(
                box_regression_per_image, targets_per_image, anchors_per_image, matched_idxs_per_image
            )
            total_box_regression_list.append(decode_box_regression_per_image)
            total_target_regression_list.append(matched_gt_boxes_per_image)

        total_box_regression = torch.cat(total_box_regression_list, dim=0)
        total_target_regression = torch.cat(total_target_regression_list, dim=0)

        if total_box_regression.shape[0] == 0:
            # if there is no training sample.
            losses = torch.tensor(0.0)
            return losses

        losses = self.box_loss_func(total_box_regression, total_target_regression).to(total_box_regression.dtype)

        return losses
    
    def get_cls_train_sample_per_image(
        self, cls_logits_per_image: Tensor, targets_per_image: dict[str, Tensor], matched_idxs_per_image: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Get samples from one image for classification losses computation.

        Args:
            cls_logits_per_image: classification logits for one image, (sum(HWA), self.num_classes)
            targets_per_image: a dict with at least two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            matched_idxs_per_image: matched index, Tensor sized (sum(HWA),) or (sum(HWDA),)
                Suppose there are M gt boxes. matched_idxs_per_image[i] is a matched gt index in [0, M - 1]
                or a negative value indicating that anchor i could not be matched.
                BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2

        Return:
            paired predicted and GT samples from one image for classification losses computation
        """

        if torch.isnan(cls_logits_per_image).any() or torch.isinf(cls_logits_per_image).any():
            if torch.is_grad_enabled():
                raise ValueError("NaN or Inf in predicted classification logits.")
            else:
                warnings.warn("NaN or Inf in predicted classification logits.")

        foreground_idxs_per_image = matched_idxs_per_image >= 0

        num_foreground = int(foreground_idxs_per_image.sum())
        num_gt_box = targets_per_image[self.target_box_key].shape[0]

        # if self.debug:
        #     print(f"Number of positive (matched) anchors: {num_foreground}; Number of GT box: {num_gt_box}.")
        #     if num_gt_box > 0 and num_foreground < 2 * num_gt_box:
        #         print(
        #             f"Only {num_foreground} anchors are matched with {num_gt_box} GT boxes. "
        #             "Please consider adjusting matcher setting, anchor setting,"
        #             " or the network setting to change zoom scale between network output and input images."
        #         )

        # create the target classification with one-hot encoding
        gt_classes_target = torch.zeros_like(cls_logits_per_image)  # (sum(HW(D)A), self.num_classes)
        # print(f"foreground_idxs_per_image dtype: {foreground_idxs_per_image.dtype}")
        # print(f"matched_idxs_per_image dtype: {matched_idxs_per_image.dtype}")
        # print(f"targets_per_image[{self.target_label_key}] dtype: {targets_per_image[self.target_label_key].dtype}")

        gt_classes_target[
            foreground_idxs_per_image,  # fg anchor idx in
            targets_per_image[self.target_label_key][
                matched_idxs_per_image[foreground_idxs_per_image]
            ],  # fg class label
        ] = 1.0

        if self.fg_bg_sampler is None:
            # if no balanced sampling
            valid_idxs_per_image = matched_idxs_per_image != self.proposal_matcher.BETWEEN_THRESHOLDS
        else:
            # The input of fg_bg_sampler: list of tensors containing -1, 0 or positive values.
            # Each tensor corresponds to a specific image.
            # -1 values are ignored, 0 are considered as negatives and > 0 as positives.

            # matched_idxs_per_image (Tensor[int64]): an N tensor where N[i] is a matched gt in
            # [0, M - 1] or a negative value indicating that prediction i could not
            # be matched. BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
            if isinstance(self.fg_bg_sampler, HardNegativeSampler):
                max_cls_logits_per_image = torch.max(cls_logits_per_image.to(torch.float32), dim=1)[0]
                sampled_pos_inds_list, sampled_neg_inds_list = self.fg_bg_sampler(
                    [matched_idxs_per_image + 1], max_cls_logits_per_image
                )
            elif isinstance(self.fg_bg_sampler, BalancedPositiveNegativeSampler):
                sampled_pos_inds_list, sampled_neg_inds_list = self.fg_bg_sampler([matched_idxs_per_image + 1])
            else:
                raise NotImplementedError(
                    "Currently support torchvision BalancedPositiveNegativeSampler and monai HardNegativeSampler matcher. "
                    "Other types of sampler not supported. "
                    "Please override self.get_cls_train_sample_per_image(*) for your own sampler."
                )

            sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds_list, dim=0))[0]
            sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds_list, dim=0))[0]
            valid_idxs_per_image = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        return cls_logits_per_image[valid_idxs_per_image, :], gt_classes_target[valid_idxs_per_image, :]

    def get_box_train_sample_per_image(
        self,
        box_regression_per_image: Tensor,
        targets_per_image: dict[str, Tensor],
        anchors_per_image: Tensor,
        matched_idxs_per_image: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Get samples from one image for box regression losses computation.

        Args:
            box_regression_per_image: box regression result for one image, (sum(HWA), 2*self.spatial_dims)
            targets_per_image: a dict with at least two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors_per_image: anchors of one image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            matched_idxs_per_image: matched index, sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            paired predicted and GT samples from one image for box regression losses computation
        """

        if torch.isnan(box_regression_per_image).any() or torch.isinf(box_regression_per_image).any():
            if torch.is_grad_enabled():
                raise ValueError("NaN or Inf in predicted box regression.")
            else:
                warnings.warn("NaN or Inf in predicted box regression.")

        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        num_gt_box = targets_per_image[self.target_box_key].shape[0]

        # if no GT box, return empty arrays
        if num_gt_box == 0:
            return box_regression_per_image[0:0, :], box_regression_per_image[0:0, :]

        # select only the foreground boxes
        # matched GT boxes for foreground anchors
        matched_gt_boxes_per_image = targets_per_image[self.target_box_key][
            matched_idxs_per_image[foreground_idxs_per_image]
        ].to(box_regression_per_image.device)
        # predicted box regression for foreground anchors
        box_regression_per_image = box_regression_per_image[foreground_idxs_per_image, :]
        # foreground anchors
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

        # encode GT boxes or decode predicted box regression before computing losses
        matched_gt_boxes_per_image_ = matched_gt_boxes_per_image
        box_regression_per_image_ = box_regression_per_image
        if self.encode_gt:
            matched_gt_boxes_per_image_ = self.box_coder.encode_single(matched_gt_boxes_per_image_, anchors_per_image)
        if self.decode_pred:
            box_regression_per_image_ = self.box_coder.decode_single(box_regression_per_image_, anchors_per_image)

        return box_regression_per_image_, matched_gt_boxes_per_image_
    
    def seg_setting(self, model, action, *args, **kwargs):
        if action == 'init':
            # loss, optimizer, lr_scheduler
            self.seg_loss = DiceLoss(
                to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6
            )
            self.seg_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            self.seg_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.seg_optimizer, T_max=self.num_epoch
            )
        elif action == 'execute':
            inputs, targets = args
            outputs = model(inputs, task = "segmentation")
            loss = self.seg_loss(outputs, targets)
            if model.training:
                loss.backward()
                self.seg_optimizer.step()
                self.seg_optimizer.zero_grad()
                return loss
            else:
                return loss, outputs

    def cls_setting(self, model, action, *args, **kwargs):
        if action == 'init':
            # loss, optimizer
            self.cls_loss = FocalLoss(reduction="mean", gamma=2.8)
            self.cls_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        elif action == 'execute':
            inputs, targets = args
            outputs = model(inputs, task = "classification")
            loss = self.cls_loss(outputs, targets)
            if model.training:
                loss.backward()
                self.cls_optimizer.step()
                self.cls_optimizer.zero_grad()
                return loss
            else:
                return loss, outputs


    def det_setting(self, model, action, *args, **kwargs):
        if action == 'init':
            self.matching_batch = matching_batch
            self.box_utils = box_utils
            self.target_box_key = "box"
            self.target_label_key = "box_label"
            self.pred_score_key = self.target_label_key + "_scores" 
            self.spatial_dims = self.get_attribute_from_network(model, "spatial_dims", default_value=3)
            self.cls_key = self.get_attribute_from_network(model, "cls_key", default_value="classification")
            self.box_reg_key = self.get_attribute_from_network(model, "box_reg_key", default_value="box_regression")
            self.anchor_generator = AnchorGeneratorWithAnchorShape(
                feature_map_scales=[2**l for l in range(1,5)],
                base_anchor_shapes=[[45,58,52],[40,65,58],[42,58,55],[30,60,40],[70,70,60]],
            )
            self.num_anchors_per_loc = self.anchor_generator.num_anchors_per_location()[0]
            self.anchors: list[Tensor] | None = None
            box_overlap_metric: Callable = box_iou
            self.box_overlap_metric = box_overlap_metric
            self.fg_bg_sampler = None 
            self.set_atss_matcher(num_candidates=4, center_in_gt=False)
            self.box_coder = BoxCoder(weights=(1.0,) * 2 * 3)
            self.box_selector = BoxSelector(
                box_overlap_metric=self.box_overlap_metric,
                score_thresh=0.02,
                topk_candidates_per_level=1000,
                nms_thresh=0.22,
                detections_per_img=1,
                apply_sigmoid=True,
            )
            
            self.cls_loss_func =FocalLoss(reduction="mean", gamma=2.0)
            self.set_box_regression_loss(
                torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"), encode_gt=True, decode_pred=False
            )  
            self.det_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
            self.det_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.det_optimizer, T_max=self.num_epoch)
 
        elif action == 'execute':
            images, targets = args
            head_outputs = model(images, task = "detection")
            # Generate anchors and store it in self.anchors
            self.generate_anchors(images, head_outputs)
            num_anchor_locs_per_level = [x.shape[2:].numel() for x in head_outputs[self.cls_key]]
            # Reshape and concatenate head_outputs values from List[Tensor] to Tensor
            for key in [self.cls_key, self.box_reg_key]:
                head_outputs[key] = self._reshape_maps(head_outputs[key])
            # If during training, return losses
            if model.training:
                losses = self.compute_loss(head_outputs, targets, self.anchors, num_anchor_locs_per_level)
                loss = losses[self.cls_key] + losses[self.box_reg_key]
                loss.backward()
                self.det_optimizer.step()
                self.det_optimizer.zero_grad()
                return loss, losses

            else:
                losses = self.compute_loss(head_outputs, targets, self.anchors, num_anchor_locs_per_level)
                loss = losses[self.cls_key] + losses[self.box_reg_key]
                detections = self.postprocess_detections(
                        head_outputs, self.anchors, [[96,96,96]], num_anchor_locs_per_level  # type: ignore
                    )
                return detections, loss, losses

    def multi_setting(self, model, action, optimizer, *args, **kwargs):
        if action == 'init':
            self.matching_batch = matching_batch
            self.box_utils = box_utils
            self.target_box_key = "box"
            self.target_label_key = "box_label"
            self.pred_score_key = self.target_label_key + "_scores" 
            self.spatial_dims = self.get_attribute_from_network(model, "spatial_dims", default_value=3)
            self.cls_key = self.get_attribute_from_network(model, "cls_key", default_value="classification")
            self.box_reg_key = self.get_attribute_from_network(model, "box_reg_key", default_value="box_regression")
            self.anchor_generator = AnchorGeneratorWithAnchorShape(
                feature_map_scales=[2**l for l in range(1,5)],
                base_anchor_shapes=[[45,58,52],[40,65,58],[42,58,55],[30,60,40],[70,70,60]],
            )
            self.num_anchors_per_loc = self.anchor_generator.num_anchors_per_location()[0]
            self.anchors: list[Tensor] | None = None
            box_overlap_metric: Callable = box_iou
            self.box_overlap_metric = box_overlap_metric
            self.fg_bg_sampler = None 
            self.set_atss_matcher(num_candidates=4, center_in_gt=False)
            self.box_coder = BoxCoder(weights=(1.0,) * 2 * 3)
            self.box_selector = BoxSelector(
                box_overlap_metric=self.box_overlap_metric,
                score_thresh=0.02,
                topk_candidates_per_level=1000,
                nms_thresh=0.22,
                detections_per_img=1,
                apply_sigmoid=True,
            )
            # Detection loss (Including classification loss and box regression loss)
            self.cls_loss_func =FocalLoss(reduction="mean", gamma=2.0)
            self.set_box_regression_loss(
                torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"), encode_gt=True, decode_pred=False
            )  
            self.seg_loss = DiceLoss( # Segmentation loss
                to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6
            )
            self.cls_loss = FocalLoss(reduction="mean", gamma=2.8) # Classification loss
            
            self.multi_optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': 1e-4},
                {'params': model.seg_decoder.parameters(), 'lr': 1e-4},
                {'params': model.det_decoder.parameters(), 'lr': 1e-5},
                {'params': model.cls_decoder.parameters(), 'lr': 1e-5},
            ])
            self.multi_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.multi_optimizer, T_max=self.num_epoch
            )
        elif action == 'execute':
            input, seg_targets, cls_targets, det_targets, step = args
            seg_outputs, cls_outputs, det_outputs = model(input)

            seg_loss = self.seg_loss(seg_outputs, seg_targets)
            cls_loss = self.cls_loss(cls_outputs, cls_targets)
            
            # Detection Loss
            # Generate anchors and store it in self.anchors
            self.generate_anchors(input, det_outputs)
            num_anchor_locs_per_level = [x.shape[2:].numel() for x in det_outputs[self.cls_key]]
            # Reshape and concatenate head_outputs values from List[Tensor] to Tensor
            for key in [self.cls_key, self.box_reg_key]:
                det_outputs[key] = self._reshape_maps(det_outputs[key])
            # If during training, return losses
            if model.training:
                det_losses = self.compute_loss(
                    det_outputs, det_targets, self.anchors, num_anchor_locs_per_level
                )
                det_loss = det_losses[self.cls_key] + det_losses[self.box_reg_key]

            else:
                det_losses = self.compute_loss(det_outputs, det_targets, self.anchors, num_anchor_locs_per_level)
                det_loss = det_losses[self.cls_key] + det_losses[self.box_reg_key]
                det_outputs = self.postprocess_detections(
                        det_outputs, self.anchors, [[96,96,96]], num_anchor_locs_per_level  # type: ignore
                    )
            
            if model.training:
                if optimizer == "GradNorm":
                    loss_list = torch.stack([seg_loss, cls_loss, det_loss])
                    lr2 = 0.001
                    alpha=0.12
                    log_weights = []
                    log_loss = []
                    # initialization
                    if step == 1:
                        # init weights
                        self.weights = torch.ones_like(loss_list)
                        self.weights = torch.nn.Parameter(self.weights)
                        self.T = self.weights.sum().detach() # sum of weights
                        # set optimizer for weights
                        self.optimizer2 = torch.optim.Adam([self.weights], lr=lr2)
                        # set L(0)
                        self.l0 = loss_list.detach()
                    # compute the weighted loss
                    weighted_loss = self.weights @ loss_list
                    # clear gradients of network
                    self.multi_optimizer.zero_grad()
                    # backward pass for weigthted task loss
                    weighted_loss.backward(retain_graph=True)
                    # compute the L2 norm of the gradients for each task
                    gw = []
                    layer = model.seg_decoder
                    for param in layer.parameters():
                        param.requires_grad = True
                    for i in range(len(loss_list)):
                        dl = torch.autograd.grad(self.weights[i]*loss_list[i], list(layer.parameters())[-1], retain_graph=True, create_graph=True)[0]
                        gw.append(torch.norm(dl))
                    gw = torch.stack(gw)
                    # compute loss ratio per task
                    loss_ratio = loss_list.detach() / self.l0
                    # compute the relative inverse training rate per task
                    rt = loss_ratio / loss_ratio.mean()
                    # compute the average gradient norm
                    gw_avg = gw.mean().detach()
                    # compute the GradNorm loss
                    constant = (gw_avg * rt ** alpha).detach()
                    gradnorm_loss = torch.abs(gw - constant).sum()
                    # clear gradients of weights
                    self.optimizer2.zero_grad()
                    # backward pass for GradNorm
                    gradnorm_loss.backward()
                    # weight for each task
                    log_weights.append(self.weights.detach().cpu().numpy().copy())
                    # task normalized loss
                    log_loss.append(loss_ratio.detach().cpu().numpy().copy())
                    # update model weights
                    self.multi_optimizer.step()
                    # update loss weights
                    self.optimizer2.step()
                    # renormalize weights
                    self.weights = (self.weights / self.weights.sum() * self.T).detach()
                    self.weights = torch.nn.Parameter(self.weights)
                    self.optimizer2 = torch.optim.Adam([self.weights], lr=lr2)
                    total_loss = det_loss + seg_loss + cls_loss
                    print(f"Weights: {self.weights.tolist()}")
                    return total_loss, seg_loss, cls_loss, det_loss
                
                elif optimizer == "MGDA":
                    tasks = ['seg', 'cls', 'det']
                    loss_fn = {
                        'seg': self.seg_loss,
                        'cls': self.cls_loss,
                        'det': self.compute_loss,
                    }
                    outputs = {
                        'seg': seg_outputs,
                        'cls': cls_outputs,
                        'det': det_outputs
                    }
                    labels = {
                        'seg': seg_targets,
                        'cls': cls_targets,
                        'det': det_targets
                    }
                    loss_data = {}
                    grads = {}
                    scale = {}
                    total_loss = 0
                    for t in tasks:
                        if t == "det":
                            loss = loss_fn[t](outputs[t], labels[t], self.anchors, num_anchor_locs_per_level)
                            loss = loss[self.cls_key] + loss[self.box_reg_key]
                        else:
                            loss = loss_fn[t](outputs[t], labels[t])
                        loss_data[t] = loss.item()
                        loss.backward(retain_graph=True)
                        
                        # Store gradients
                        grads[t] = []
                        for param in model.encoder.parameters():  
                            if param.grad is not None:
                                grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))
                    
                    # Normalize all gradients
                    gn = gradient_normalizers(grads, loss_data, 'l2')  
                    for t in tasks:
                        for gr_i in range(len(grads[t])):
                            grads[t][gr_i] = grads[t][gr_i] / gn[t]
                    
                    # Frank-Wolfe iteration to compute scales.
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                    for i, t in enumerate(tasks):
                        scale[t] = float(sol[i])
                        print(f"Task {t}, scale is {scale[t]}") 
                        
                    # Scaled back-propagation  
                    self.multi_optimizer.zero_grad()
                    for i, t in enumerate(tasks):
                        if t == "det":
                            losses_t = loss_fn[t](outputs[t], labels[t], self.anchors, num_anchor_locs_per_level)
                            loss_t = losses_t[self.cls_key] + losses_t[self.box_reg_key]
                        else:
                            loss_t = loss_fn[t](outputs[t], labels[t])
                        loss_data[t] = loss_t.item()
                        if i > 0:
                            loss = loss + scale[t] * loss_t
                        else:
                            loss = scale[t] * loss_t
                        total_loss = loss
                    loss.backward()
                    self.multi_optimizer.step()
                    return total_loss, loss_data['seg'], loss_data['cls'], loss_data['det']

            else:
                return seg_loss, cls_loss, det_loss, seg_outputs, cls_outputs, det_outputs