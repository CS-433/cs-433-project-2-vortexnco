import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import random
import torch

from PIL import Image
from itertools import product
from helpers import get_DataLoaders, get_image_file
from AvailableRooftopDataset import AvailableRooftopDataset
from torchvision import transforms
from model.unet_model import UNet

dataFolder = "../data"
imageFolder = os.path.join(dataFolder, "images")
labelFolder = os.path.join(dataFolder, "labels")
test_filename = "DOP25_LV03_1301_11_2015_1_15_497500.0_119062.5.png"

# POS = 255
# NEG = 0
# true_value = 255
map_rgb = {
    "tp" : [0, 255, 0]
    "tn" : [0, 0, 0]
    "fp" : [255, 0, 0]
    "fn" : [255, 215, 0]
}

# If color values are binary
COMPARE_MAP_01 = {
    (2,0)  : "tp",
    (0,0)  : "tn",
    (1,1)  : "fp",
    (1,-1) : "fn"
}

# If color values are 3 bytes
# COMPARE_MAP_uint8 = {
#     (254, 0): true_positive,
#     (0, 0): true_negative,
#     (255, 255): false_positive,
#     (255, 1): false_negative,
# }


def compare_labels(true_label, predicted_label):
    """Outputs an array annotated as TP, FP, TN or FN"""
    height, width = true_label.shape
    comp_array = np.array([predicted_label + true_label, predicted_label - true_label])
    f = lambda i, j: COMPARE_MAP_01[tuple(comp_array[:, i, j])]

    result = np.fromfunction(f, (height, width), dtype=str)
    return result
    #result = np.empty((3, height, width), dtype=int)
    #for i, j in product(range(height), range(height)):
        #result[:, i, j] = f(i, j)

    #return result.transpose((1,2,0))


def show_label_comparison(true_label, predicted_label):
    """
    Plots an array annotated with TP, FP, TN and FN

    Parameters
    ----------
    true_label : ndarray of 1s and 0s
        True label.
    predicted_label : ndarray of 1s and 0s
        Prediction from the model.

    Returns
    -------
    None.
    """
    
    comparison = compare_labels(true_label, predicted_label)
    # Recover comparison array and convert it to RGB
    comparison_rgb = np.empty((height, width, 3), dtype=int)
    for i, j in product(range(height), range(width)):
        comparison_rgb[i, j, :] = f(i, j)

    plt.imshow(comparison_rgb)
    TP = mpatches.Patch(color="green", label="TP")
    TN = mpatches.Patch(color="black", label="TN")
    FP = mpatches.Patch(color="red", label="FP")
    FN = mpatches.Patch(color=[255 / 255, 215 / 255, 0], label="FN")
    plt.legend(handles=[TP, FP, TN, FN], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

def post_processing_tuning(
        batch_size : int = 32,
        train_percentage : float = 0.7,
        validation_percentage : float = 0.15,
        test_percentage : float = 0.15,
        dir_data : str ="/raid/machinelearning_course/data/",
        use_noPV : bool = False,
        prop_noPV : float = 0.0,
        min_rescale_images : float = 0.6,
        dir_for_model_parameters : str = "../saved_models",
        filename_model_parameters_to_load : str = None,
    ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is {}available.".format("" if torch.cuda.is_available() else "NOT "))

    # Instantiate the dataset
    roof_dataset = AvailableRooftopDataset(
        dir_PV = os.path.join(dir_data, "images"), 
        dir_noPV = os.path.join(dir_data, "noPV"), 
        dir_labels = os.path.join(dir_data, "labels"),  
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(250, scale=(min_rescale_images, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        use_noPV = use_noPV,
        prop_noPV = prop_noPV 
    )

    # Create the DataLoaders (if used with default 0.7 / 0.15 / 0.15 will be the same as the ones used during training)
    _, roof_dataloader_validation, roof_dataloader_test = get_DataLoaders(
        roof_dataset,
        train_percentage,
        validation_percentage,
        test_percentage,
        batch_size
        )
    
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device)
    path_model_parameters_to_load = os.path.join(dir_for_model_parameters, filename_model_parameters_to_load)
    model.load_state_dict(torch.load(path_model_parameters_to_load))
    
    model.eval()
    
    #torch.no_grad() in order not to compute gradients (better performance and memory)
    with torch.no_grad():
        for batch_x, batch_y in roof_dataloader_validation:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(
                device, dtype=torch.float32
            )
            
            #shape is BxHxWxC usually (batch_size, 250, 250, 3)
            image_numpy = batch_x.cpu().numpy().transpose((0,2,3,1))
            
            #output have not yet passed a sigmoid layer when exiting the model
            #shape is (batch_size, 1, 250, 250)
            output = model(batch_x)
            
            #shape of output_numpy is (batch_size, 250, 250)
            output_numpy = np.squeeze(output.cpu().numpy())
            
            #sigmoid function to get the probabilities between 0 and 1
            #shape of probabilities_numpy is (batch_size, 250, 250)
            probabilities_numpy = 1/(1 + np.exp(-output_numpy)) 
            
            threshold_true_label = 0.5
            threshold_prediction = 0.9
            
            #shape of decision_numpy is (batch_size, 250, 250) 
            decision_numpy = np.where(probabilities_numpy>threshold_prediction, 1., 0.)
            
            #shape of label_numpy is (batch_size, 250, 250) 
            label_numpy = batch_y.cpu().numpy()
            # need to threshold because RandomResizedCrop transorm can change the pixel value
            # need to put back all pixels to 0 or 1
            label_numpy = np.where(label_numpy>threshold_true_label, 1., 0.)
            
            ##########
            #Insert your code here for validation etc... according to thresholds
            # you can do the same after with the test_set
            #example below of some plots:
            for i in range(batch_size):
                plt.imshow(image_numpy)
                plt.show()
                plt.imshow(label_numpy)
                plt.show()
                plt.imshow(decision_numpy)
                plt.show()
                show_label_comparison(label_numpy, decision_numpy)
                print("Exiting all loops to avoid printing all images")
                break
            break
            

def eval_model(model, val_set):


if __name__ == "__main__":
    post_processing_tuning(filename_model_parameters_to_load = "some_cool_model.pt")
