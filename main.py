import argparse
import os
import time
from train import Trainer
from lib.data_loader import train_transforms, val_transforms, load_decathlon_datalist
from monai.data import CacheDataset, DataLoader
from torch.backends import cudnn
import random
import warnings
from sklearn.model_selection import KFold
import numpy as np
from monai.data import Dataset

def load_data_indices(full_data, index):
    """Helper function to fetch data based on indices."""
    return [full_data[i] for i in index]

def main(config):
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)
    
    #Mkdir for results
    root_dir ="./result"
    if config.task == "segmentation": 
        root_dir = os.path.join(root_dir, "train_seg")
    elif config.task == "detection": 
        root_dir = os.path.join(root_dir, "train_det")
    elif config.task == "classification":
        root_dir = os.path.join(root_dir, "train_cls")
    else:
        root_dir = os.path.join(root_dir, "train_multi")
    date = str(time.strftime("%m_%d_%H_%M", time.localtime()))
    folder_name = f"brain_tumor_{config.multi_opt}_{date}"
    root_dir = os.path.join(root_dir, folder_name)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    g = open(os.path.join(root_dir, 'fold_indices.txt'), 'a+')
    
    #BraST2018
    full_files = load_decathlon_datalist(
            config.json_file,
            is_segmentation=True,
            data_list_key="training",
            base_dir=config.data_dir,
            seg_label_dir=config.seg_label_dir,
        )

    #Utilize K Fold Cross Validation
    if config.use_k_fold:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = kf.split(full_files)
        for i, (train_index, val_index) in enumerate(folds):
            train_files = load_data_indices(full_files, train_index)
            val_files = load_data_indices(full_files, val_index)
            # print(f"Fold {i+1}")
            # print("Training indices:", train_index)
            # print("Validation indices:", val_index)           
            g.write(f"-----------------------Fold {i+1}-------------------------\n")
            g.write(f"Training indices: {train_index}\n")
            g.write(f"Validation indices: {val_index}\n")

            g.write("Training data samples:\n")
            for idx, sample in enumerate(train_files):
                g.write(f"Index {idx}: {sample}\n")
            g.write("Validation data samples:\n")
            for idx, sample in enumerate(val_files):
                g.write(f"Index {idx}: {sample}\n")

            fold_dir = os.path.join(root_dir, f"fold_{i + 1}")
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            
            train_ds = Dataset(
                    data=train_files,
                    transform=train_transforms,
                    )
            val_ds = Dataset(
                    data=val_files,
                    transform=val_transforms,
                )
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
            print(f"Train data fold {i+1} loaded")
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            test_loader = DataLoader(val_ds, batch_size=1, num_workers=1)
            print("Start training...")
            solver = Trainer(config, train_loader, val_loader, test_loader, fold_dir)
            # Start train
            if config.mode == 'train':
                solver.swin_train()
            elif config.mode == 'test':
                solver.swin_train()
    else:
        # Create a single train-test split (80% train, 20% validation)
        # train_files = full_files[:int(0.8 * len(full_files))]
        # val_files = full_files[int(0.8 * len(full_files)):]
        train_files = full_files[:int(0.1 * len(full_files))]
        val_files = full_files[int(0.9 * len(full_files)):]
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
        print("Start training...")
        solver = Trainer(config, train_loader, val_loader, test_loader, root_dir)

        # Train and sample the images
        if config.mode == 'train':
            solver.swin_train()
        elif config.mode == 'test':
            solver.swin_train()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=96)

    # training hyper-parameters
    parser.add_argument('--json_file', type=str, default='../Medical-image-analysis/utils/data_annotation.json')
    parser.add_argument('--data_dir', type=str, default='../dataset/BraTS/imageTr2018')
    parser.add_argument('--seg_label_dir', type=str, default='../dataset/BraTS/labelTr2018')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--task', type=str, default='multi', help="Specify the type of training. Options: segmentation(single-task), detection(single-task), classification(single-task), multi(multi-task).")
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--use_k_fold', default=True, help="Enable k-fold cross-validation. Default is True.")
    parser.add_argument('--multi_opt', default="GradNorm", help="Optimization strategy for multi-task learning. Options: 'GradNorm', 'MGDA'")

    config = parser.parse_args()
    main(config)
