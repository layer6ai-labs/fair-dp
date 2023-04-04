import os
import random
import PIL
from pathlib import Path
from typing import Tuple, Sequence, Any

import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

from .dataset import GroupLabelDataset
from .sample_weights import find_sample_weights


class CelebA(Dataset):
    """
    CelebA PyTorch dataset
    The built-in PyTorch dataset for CelebA is outdated.
    """

    def __init__(self, root: str, group_ratios: Sequence[int], role: str = "train", seed: int = 0):
        self.root = Path(root)
        self.role = role

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        celeb_path = lambda x: self.root / x

        role_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        splits_df = pd.read_csv(celeb_path("list_eval_partition.csv"))
        fields = ['image_id', 'Male', 'Eyeglasses']
        attrs_df = pd.read_csv(celeb_path("list_attr_celeba.csv"), usecols=fields)
        df = pd.merge(splits_df, attrs_df, on='image_id')
        df = df[df['partition'] == role_map[self.role]].drop(labels='partition', axis=1)
        df = df.replace(to_replace=-1, value=0)

        if seed:
            # Shuffle order according to seed but keep standard partition because the same person appears multiple times
            state = np.random.default_rng(seed=seed)
            df = df.sample(frac=1, random_state=state)
        
        labels = df["Male"]
        if group_ratios and (role_map[self.role] != 2):
            # don't alter the test set, refer to sample_weights.py
            label_counts = labels.value_counts(dropna=False).tolist()
            sample_weights = find_sample_weights(group_ratios, label_counts)
            print(f"Number of samples by label (before sampling) in {self.role}:")
            print(f"Female: {label_counts[0]}, Male: {label_counts[1]}")

            random.seed(seed)
            idx = [random.random() <= sample_weights[label] for label in labels]
            labels = labels[idx]
            label_counts_after = labels.value_counts(dropna=False).tolist()

            print("Number of samples by label (after sampling):")
            print(f"Female: {label_counts_after[0]}, Male: {label_counts_after[1]}")
            df = df[idx]
        
        self.filename = df["image_id"].tolist()
        # Male is 1, Female is 0
        self.y = torch.Tensor(df["Male"].values).long()
        # Wearing glasses is 1, otherwise zero
        self.z = torch.Tensor(df["Eyeglasses"].values).long()
        
        self.shape = (len(self.filename), 3, 64, 64)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img_path = (self.root / "img_align_celeba" /
                    "img_align_celeba" / self.filename[index])
        x = PIL.Image.open(img_path)
        x = self.transform(x).to(self.device)
        
        y = self.y[index].to(self.device)
        z = self.z[index].to(self.device)

        return x, y, z

    def __len__(self) -> int:
        return len(self.filename)
    
    def to(self, device):
        self.device = device
        return self


def image_tensors_to_dataset(dataset_role, images, labels):
    images = images.to(dtype=torch.get_default_dtype())
    labels = labels.long()
    # NOTE: assumed protected group is defined by labels for image dsets
    return GroupLabelDataset(dataset_role, images, labels, labels)


# Returns tuple of form `(images, labels)`.
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 1}
def get_raw_image_tensors(dataset_name, train, data_root, group_ratios=None, seed=0):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)

    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train" if train else "test", download=True)
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.labels)

    elif dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    images = images / 255.0

    if group_ratios:
        # refer to sample_weights.py
        _, label_counts = torch.unique(labels, sorted=True, return_counts=True)
        sample_weights = find_sample_weights(group_ratios, label_counts.tolist())
        print("Number of samples by label (before sampling):")
        print(label_counts)

        random.seed(seed)
        idx = [random.random() <= sample_weights[label.item()] for label in labels]
        labels = labels[idx]
        images = images[idx]
        _, label_counts_after = torch.unique(labels, sorted=True, return_counts=True)

        print("Number of samples by label (after sampling):")
        print(label_counts_after)

    return images, labels


def get_torchvision_datasets(dataset_name, data_root, seed, group_ratios, valid_fraction, flatten):
    images, labels = get_raw_image_tensors(dataset_name, train=True, data_root=data_root, group_ratios=group_ratios,
                                           seed=seed)
    if flatten:
        images = images.flatten(start_dim=1)

    perm = torch.randperm(images.shape[0])
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]

    train_dset = image_tensors_to_dataset("train", train_images, train_labels)
    valid_dset = image_tensors_to_dataset("valid", valid_images, valid_labels)

    test_images, test_labels = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    if flatten:
        test_images = test_images.flatten(start_dim=1)
    test_dset = image_tensors_to_dataset("test", test_images, test_labels)

    return train_dset, valid_dset, test_dset

def get_image_datasets_by_class(dataset_name, data_root, seed, group_ratios, valid_fraction, flatten=False):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "celeba":
        # valid_fraction and flatten ignored
        data_class = CelebA

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    train_dset = data_class(root=data_dir, group_ratios=group_ratios, role="train", seed=seed)
    valid_dset = data_class(root=data_dir, group_ratios=group_ratios, role="valid", seed=seed)
    test_dset = data_class(root=data_dir, group_ratios=group_ratios, role="test", seed=seed)

    return train_dset, valid_dset, test_dset

def get_image_datasets(dataset_name, data_root, seed, group_ratios, make_valid_loader=False, flatten=False):
    valid_fraction = 0.1 if make_valid_loader else 0

    torchvision_datasets = ["mnist", "fashion-mnist", "svhn", "cifar10"]
    
    get_datasets_fn = get_torchvision_datasets if dataset_name in torchvision_datasets else get_image_datasets_by_class

    return get_datasets_fn(dataset_name, data_root, seed, group_ratios, valid_fraction, flatten)
