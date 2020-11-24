import os
import random

from torch.utils.data import Dataset


class AvailableRooftopDataset(Dataset):
    """Available Rooftop Dataset."""

    def __init__(self, dir_images, dir_labels, transform=None):
        """
        Args:
            root_dir_images (string): Directory with all the images.
            root_dir_labels (string): Directory with all the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_images = dir_images
        self.dir_labels = dir_labels
        self.transform = transform

        # Get the list of image/label name from images/labels directory, except dotfile
        self.images_name = [image_name for image_name in os.listdir(dir_images) if image_name[0] != '.']
        self.labels_name = [label_name for label_name in os.listdir(dir_labels) if label_name[0] != '.']

        # Create an image -> label dict
        self.image_label_dict = {}

        # Iterate through all the images' name to add them in the dict
        for image_full_name in self.images_name:
            image_name, image_extension = os.path.splitext(image_full_name)

            # Find the label of the image, if there is one
            label_name_associated = None
            for label_full_name in self.labels_name:
                if image_name in label_full_name:
                    label_name_associated = label_full_name

            # If no label associated, then it should be a black label
            if (!label_name_associated):
                label_name_associated = 'DEFAULT'

            self.image_label_dict[image_full_name] = label_name_associated

        # Shuffle the images' name to avoid having an order when retrieving the images in __getitem__
        random.shuffle(self.images_name)


    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None
       