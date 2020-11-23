# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:15:20 2020

@author: Alexander
"""
# Import library
import os
from torch.utils.data import Dataset as BaseDataset

#file structure
"""
.--src--.
|       |        
|       .--loaders.py
|       .--[other src files]
|
.--data--.
|        |
|        .--PV--.
[...]    |      |
         |      .--labels--.
         |      |          |
         |      |          .--xxx_label.png
         |      |          .--[other labels]
         |      |
         |      .--xxx.png
         |      .--[other images]
         |
         .--noPV--.
                  |
                  .--yyy.png
                  .--[other images]

"""

#code adapted from https://programmersought.com/article/26334490573/
#code from https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html


#Next three lines might be done from the pipeline.py script for modularity or
#put in a file constants.py 
#TODO decide
DATA_DIR ='../data/PV' # Set according to your own path
x_train_dir = DATA_DIR
y_train_dir = os.path.join(DATA_DIR, 'labels')

 
 # Custom Dataloader
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        transformations (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_filenames = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        self.masks_filenames = [f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))]
        
        self.transform = transform
 
    
    def __getitem__(self, idx):
        raise NotImplementedError
        """if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample"""
    
    def __len__(self):
        return len(self.images_filenames)
    
if __name__ == "__main__":
    print("Hello you")
