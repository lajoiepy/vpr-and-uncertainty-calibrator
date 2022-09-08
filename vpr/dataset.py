import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ImageSequenceFolder(datasets.ImageFolder):
    """TODO
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        is_validation_set=False,
        validation_step=None,
    ):
        self.validation_step = validation_step
        self.is_validation_set = is_validation_set
        super().__init__(root=root,
                         transform=transform,
                         is_valid_file=self.is_valid_file)

    def get_image_id(self, path):
        filename = path.split("/")[-1]
        return int(filename.split(".")[0])

    def is_valid_file(
            self,
            path): 
        seq_id = self.get_image_id(path)
        if self.validation_step is not None:
            if self.is_validation_set:
                if seq_id % self.validation_step != 0:
                    return False
            else:
                if seq_id % self.validation_step == 0:
                    return False

        try:
            img = Image.open(path)
            img.verify()
        except:
            return False

        return True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is sequence ID.
        """
        path, target = self.samples[index]
        if self.is_valid_file(path):
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)

            seq_id = self.get_image_id(path)

            return sample, seq_id
        else:
            return None, None
    
    def __contains__(self, index):
        path, _ = self.samples[index]
        return self.is_valid_file(path)

default_transform = transforms.Compose([
        transforms.CenterCrop(480),
        transforms.Resize(224, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

def create_safe_dataset(image_folder):
    dataset = ImageSequenceFolder(image_folder,
                                transform=default_transform)
    return dataset

def create_safe_dataloader(dataset, batch_size=1):
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    return dataloader

def create_safe_dataloaders_split(image_folder, batch_size=None, validation_step=None):
    training_dataset = ImageSequenceFolder(image_folder,
                                           transform=default_transform,
                                           is_validation_set=False,
                                           validation_step=validation_step)

    validation_dataset = ImageSequenceFolder(image_folder,
                                             transform=default_transform,
                                             is_validation_set=True,
                                             validation_step=validation_step)

    training_dataloader = data.DataLoader(training_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
    validation_dataloader = data.DataLoader(validation_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    return training_dataset, training_dataloader, validation_dataset, validation_dataloader
