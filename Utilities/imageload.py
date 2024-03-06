import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as TF
import random

class ImageLoad(Dataset,):
    def __init__(self, image_dir, mask_dir, transform=None, mean = None, std = None, image_dimension = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mean = mean
        self.std = std
        self.image_dimension = image_dimension
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        if not img_path.endswith(('.png', '.jpg', '.jpeg')):
            return self.__getitem__(idx + 1)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))  # adjust extension if necessary

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        if self.transform:
            image, mask = self.transform(image, mask, self.mean, self.std, self.image_dimension)
        
        return image, mask