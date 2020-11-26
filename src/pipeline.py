# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:23:24 2020

@author: Alexander
"""
from unet_model import UNet
import torch
from losses import IOULoss, iou
from AvailableRooftopDataset import AvailableRooftopDataset 

def train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs, device):
    """
    Parameters
    ----------
    model : torch.nn.Module
        DESCRIPTION.
    criterion : torch.nn.modules.loss._Loss
        DESCRIPTION.
    dataset_train : torch.utils.data.DataLoader
        DESCRIPTION.
    dataset_test : torch.utils.data.DataLoader
        DESCRIPTION.
    optimizer : torch.optim.Optimizer
        DESCRIPTION.
    num_epochs : int
        DESCRIPTION.
    device : 

    Returns
    -------
    None.

    """
    
    print("Starting training")
    for epoch in range(num_epochs):
    # Train an epoch
        model.train()
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Evaluate the network (forward pass)
            model.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            # Compute the gradient
            loss.backward()
            
            # Update the parameters of the model with a gradient step
            optimizer.step()
              
            # Test the quality on the test set
            model.eval()
            accuracies_test = []
            for batch_x, batch_y in dataset_test:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Evaluate the network (forward pass)
                prediction = model(batch_x)
                accuracies_test.append(iou(prediction, batch_y))
        
        print("Epoch {} | Test IoU: {:.5f}".format(epoch, sum(accuracies_test).item()/len(accuracies_test)))

def main(num_epochs = 10, learning_rate = 1e-3, batch_size = 128):  
    
    
    # If a GPU is available (should be on Colab, we will use it)
    """
    if not torch.cuda.is_available() or :
        raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = IOULoss()
    
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    model = model.to(device)
    dataset_train, dataset_test = get_train_test_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs, device)
    
if __name__ == "__main__":
    main()
    
    