import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io

class AvailableRooftopDataset(Dataset):
    """Available Rooftop Dataset."""

    def __init__(self, dir_PV, dir_noPV, dir_labels, transform=None, use_noPV=False, prop_noPV=0.0):
        """
        Args:
            dir_images (string): Directory with all the images.
            dir_labels (string): Directory with all the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_PV = dir_PV
        self.dir_labels = dir_labels
        self.transform = transform

        # Get the list of PV_image and label names from PV and label directories (except dotfile)
        self.images_name = [image_name for image_name in os.listdir(dir_PV) if image_name[0] != "."]        
        self.labels_name = [label_name for label_name in os.listdir(dir_labels) if label_name[0] != "."]

        # Create an image -> label dict
        self.image_label_dict = {}

        # Iterate through all the PV images' name to add them in the dict
        for image_full_name in self.images_name:
            image_name, image_extension = os.path.splitext(image_full_name)

            # Find the label of the image
            label_name_associated = None
            for label_full_name in self.labels_name:
                if image_name in label_full_name:
                    label_name_associated = label_full_name

            self.image_label_dict[image_full_name] = label_name_associated

        if (use_noPV):
            self.dir_noPV = dir_noPV

            # Get the list of noPV_image from the noPV directory (except dotfile)
            self.noPV_images_name = [image_name for image_name in os.listdir(dir_noPV) if image_name[0] != "."]

            # Keep a proportion of (random) noPV images
            random.shuffle(self.noPV_images_name)
            self.noPV_images_name = self.noPV_images_name[:int(len(self.noPV_images_name) * prop_noPV)]

            # Iterate through all the noPV images' name to add them in the dict and to the images_name list
            for noPV_image_full_name in self.noPV_images_name:
                self.image_label_dict[noPV_image_full_name] = "DEFAULT"
                # Add a '-' before the name of the file to tag it as a noPV
                self.images_name.append('-' + noPV_image_full_name)

    def __len__(self):
        return len(self.image_label_dict)

    def __getitem__(self, idx):
        # In case we use random_split
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve the image (from dir_noPV if it is tagged with '-', else from dir_PV)
        image_name = self.images_name[idx]

        if (image_name[0] == '-'):
            # Remove the tag
            image_name = image_name[1:]
            image_path = os.path.join(self.dir_noPV, image_name)
        else:
            image_path = os.path.join(self.dir_PV, image_name)

        image = io.imread(image_path)

        # Retrieve the label if the image is from PV, else the label is full black
        label_name = self.image_label_dict[image_name]
        
        if label_name != "DEFAULT":
            label_path = os.path.join(self.dir_labels, label_name)
            label = io.imread(label_path)
        else:
            label = np.zeros((250, 250, 3), dtype=np.uint8)  

        # Define a seed to apply the same transforms for 'image' and 'label'
        seed = np.random.randint(2147483647)

        # Apply the transforms if any
        if self.transform:
            # Apply transforms on image (and define seed)
            torch.manual_seed(seed)
            random.seed(seed)
            image = self.transform(image)

            # Apply transforms on label (and redefine seed)
            torch.manual_seed(seed)
            random.seed(seed)
            label = self.transform(label)

        return image, label[0]