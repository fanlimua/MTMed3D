from __future__ import annotations
from monai.utils import first
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandFlipd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
    EnsureTyped,
    NormalizeIntensityd,
    Resized,
    MapTransform,
    CenterSpatialCropd,
    RandSpatialCropd,
    CropForeground,
    RandRotated,
    DeleteItemsd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    Crop,
    MapTransform,
    CenterSpatialCrop,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    SpatialCrop,
    SpatialCropBox,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
    ConvertBoxModed,
    StandardizeEmptyBoxd,
    RandCropBoxByPosNegLabeld,
)
import numpy as np
from monai.networks.layers import Norm
from monai.data import Dataset
import os
import torch
import matplotlib.pyplot as plt
from monai.data import DataLoader
from pathlib import Path
from typing import overload
from monai.config import PathLike
import json
import matplotlib.patches as patches
import re
from collections.abc import Sequence
from monai.config import KeysCollection
from monai.utils import ensure_tuple
from collections.abc import Hashable, Mapping

class CenterCropBox(MapTransform):
    def __init__(
        self,
        image_keys: Sequence[str],
        box_keys: str,
        label_keys: KeysCollection,
        spatial_size: Sequence[int] | int,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            image_keys (Sequence[str]): The keys corresponding to the images to be cropped.
            box_keys (str): The key corresponding to the bounding boxes to be cropped.
            spatial_size (Sequence[int] | int): The size of the region of interest (ROI) to crop.
            allow_missing_keys (bool): Whether to allow missing keys. Default is False.
        """
        self.image_keys = image_keys
        self.box_keys = box_keys
        self.spatial_size = spatial_size
        box_keys_tuple = ensure_tuple(box_keys)
        if len(box_keys_tuple) != 1:
            raise ValueError(
                "Please provide a single key for box_keys.\
                All label_keys are attached to this box_keys."
            )
        self.box_keys = box_keys_tuple[0]
        self.label_keys = ensure_tuple(label_keys)
        super().__init__(keys=self.image_keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        d = dict(data)
        image_size = d[self.image_keys[0]].shape[1:]  # assuming channel-first format
        
        # Ensure the spatial size is a tuple
        spatial_size = self.spatial_size if isinstance(self.spatial_size, tuple) else (self.spatial_size,) * len(image_size)
        roi_center = [dim_size // 2 for dim_size in image_size]
        
        # Compute the cropping slices
        cropper = SpatialCrop(roi_center=roi_center, roi_size=spatial_size)
        crop_slices = cropper.slices
        
        # Crop images
        for image_key in self.image_keys:
            d[image_key] = cropper(d[image_key])
        
        # Crop bounding boxes
        labels = [d[label_key] for label_key in self.label_keys]
        boxcropper = SpatialCropBox(roi_slices=crop_slices)
        d[self.box_keys], cropped_labels = boxcropper(d[self.box_keys], labels)
        
        return d

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
class ConvertToMultiChannelBasedOnBratsClassesd2018(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    BraST2018
    label 2 the peritumoral edema, 
    label 4 GD-enhancing tumor, 
    label 1 the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
        
    Converts to multi-channel format:
        - TC (Tumor core): (label 1 or 4)
        - WT (Whole tumor): (label 1, 2, or 4)
        - ET (Enhancing tumor): (label 4)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 4 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 4 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 4), d[key] == 1))
            # label 4 is ET
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d

@overload
def _compute_path(base_dir: PathLike, element: PathLike, check_path: bool = False) -> str:
    ...

@overload
def _compute_path(base_dir: PathLike, element: list[PathLike], check_path: bool = False) -> list[str]:
    ...

def _compute_path(base_dir, element, check_path=False):

    def _join_path(base_dir: PathLike, item: PathLike):
        result = os.path.normpath(os.path.join(base_dir, item))
        if check_path and not os.path.exists(result):
            # if not an existing path, don't join with base dir
            return f"{item}"
        return f"{result}"

    if isinstance(element, (str, os.PathLike)):
        return _join_path(base_dir, element)
    if isinstance(element, list):
        for e in element:
            if not isinstance(e, (str, os.PathLike)):
                return element
        return [_join_path(base_dir, e) for e in element]
    return element


def _append_paths(base_dir: PathLike, seg_label_dir: PathLike, is_segmentation: bool, items: list[dict]) -> list[dict]:

    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Every item in items must be a dict but got {type(item).__name__}.")
        for k, v in item.items():
            if k == "image" or is_segmentation and k == "box_label":
                item[k] = _compute_path(base_dir, v, check_path=False)
            elif k == "seg_label":
                item[k] = _compute_path(seg_label_dir, v, check_path=False)
            else:
                # for other items, auto detect whether it's a valid path
                item[k] = _compute_path(base_dir, v, check_path=True)
    return items


def load_decathlon_datalist(
    data_list_file_path: PathLike,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir: PathLike | None = None,
    seg_label_dir: PathLike | None = None,
) -> list[dict]:
 
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file() and seg_label_dir.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
    expected_data = json_data[data_list_key]
    if data_list_key == "test" and not isinstance(expected_data[0], dict):
        # decathlon datalist may save the test images in a list directly instead of dict
        expected_data = [{"image": i} for i in expected_data]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    return _append_paths(base_dir, seg_label_dir, is_segmentation, expected_data)

#Setup transforms for training and validation
train_transforms = Compose(
        [
            LoadImaged(keys=["image", "seg_label"], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "seg_label"]),
            EnsureTyped(keys=["box","box_label"], dtype=torch.float32),
            ConvertToMultiChannelBasedOnBratsClassesd2018(keys="seg_label"),
            StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
            Orientationd(keys=["image", "seg_label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Spacingd(
                keys=["image", "seg_label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CenterCropBox(
                image_keys=["image","seg_label"],
                box_keys=["box"],
                label_keys=["box_label"],
                # spatial_size=(150, 150, 150),
                # spatial_size=(160, 160, 128),
                # spatial_size=(128, 128, 128),
                spatial_size=(96, 96, 96),
                # spatial_size=(64, 64, 64),
            ),
            RandFlipBoxd(
                image_keys=["image", "seg_label"],
                box_keys=["box"],
                box_ref_image_keys=["image"],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipBoxd(
                image_keys=["image", "seg_label"],
                box_keys=["box"],
                box_ref_image_keys=["image"],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipBoxd(
                image_keys=["image", "seg_label"],
                box_keys=["box"],
                box_ref_image_keys=["image"],
                prob=0.5,
                spatial_axis=2,
            ),
            RandRotateBox90d(
                image_keys=["image", "seg_label"],
                box_keys=["box"],
                box_ref_image_keys=["image"],
                prob=0.75,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["box_label"], dtype=torch.long),
        ]
    )

val_transforms = Compose(
        [
            LoadImaged(keys=["image", "seg_label"], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "seg_label"]),
            EnsureTyped(keys=["box","box_label"], dtype=torch.float32),
            ConvertToMultiChannelBasedOnBratsClassesd2018(keys="seg_label"),
            StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
            Orientationd(keys=["image", "seg_label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Spacingd(
                keys=["image", "seg_label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ), 
            CenterCropBox(
                image_keys=["image","seg_label"],
                box_keys=["box"],
                label_keys=["box_label"],
                # spatial_size=(160, 160, 128),
                # spatial_size=(128, 128, 128),
                spatial_size=(96, 96, 96),
                # spatial_size=(64, 64, 64),
            ),
            # CenterSpatialCropd(
            #     keys=["image","seg_label"],
            #     roi_size=(128, 128, 128),
            #     # roi_size=(160, 160, 128),
            # ),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["box_label"], dtype=torch.long),
        ]
    )

test_transforms = Compose(
        [
            LoadImaged(keys=["image", "seg_label"], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "seg_label"]),
            EnsureTyped(keys=["box","box_label"], dtype=torch.float32),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys="seg_label"),
            ConvertToMultiChannelBasedOnBratsClassesd2018(keys="seg_label"),
            StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
            Orientationd(keys=["image", "seg_label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Spacingd(
                keys=["image", "seg_label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(
                keys=["image","seg_label"],
                roi_size=(128, 128, 128),
            ),
            Resized(keys=["image", "seg_label"], spatial_size=(96,96,96), mode=("bilinear", "nearest")),
        ]
    )

det_post_transforms = Compose(
        [
            ClipBoxToImaged(
                box_keys=["pred_box"],
                label_keys=["pred_label", "pred_score"],
                box_ref_image_keys="image",
                remove_empty=False,
            ),
            AffineBoxToWorldCoordinated(
                box_keys=["pred_box"],
                box_ref_image_keys="image",
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=True,
            ),
            DeleteItemsd(keys=["image"]),
        ]
    )

test_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "seg_label"], meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "seg_label"]),
        EnsureTyped(keys=["box","box_label"], dtype=torch.float32),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg_label"),
        StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
        Orientationd(keys=["image", "seg_label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "seg_label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CenterSpatialCropd(
            keys=["image","seg_label"],
            roi_size=(128, 128, 128),
        ),
        Resized(keys=["image", "seg_label"], spatial_size=(96,96,96), mode=("bilinear", "nearest")),
    ]
)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
    ]
)

#BraST2018
data_json_file_path = "/home/fan/project/Medical-image-analysis/utils/data_annotation.json"
data_dir = "/home/fan/project/dataset/BraTS/imageTr2018"
seg_label_dir = "/home/fan/project/dataset/BraTS/labelTr2018"
test_dir = "/home/fan/project/dataset/Task01_BrainTumour/imagesTs"

train_files = load_decathlon_datalist(
        data_json_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=data_dir,
        seg_label_dir=seg_label_dir,
    )
test_files = load_decathlon_datalist(
        data_json_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=test_dir,
        seg_label_dir=seg_label_dir,
    )
full_files = load_decathlon_datalist(
        data_json_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=data_dir,
        seg_label_dir=seg_label_dir,
    )