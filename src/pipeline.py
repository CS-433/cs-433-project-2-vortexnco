import torch

from AvailableRooftopDataset import AvailableRooftopDataset 
from torch.utils.data import DataLoader
from model.unet_model import UNet
from losses import GeneralLoss, jaccard_loss, jaccard_distance_loss, DiceLoss
import torch.nn as nn
    

def train(model, criterion, dataloader_train, dataloader_test, optimizer, num_epochs, device):
    """
    Parameters
    ----------
    model : torch.nn.Module
    criterion : torch.nn.modules.loss._Loss
    dataloader_train : torch.utils.data.DataLoader
    dataloader_test : torch.utils.data.DataLoader
    optimizer : torch.optim.Optimizer
    num_epochs : int
    device : 

    Returns
    -------
    None
    """
    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        for sample_batched in dataloader_train:
            batch_x, batch_y = sample_batched['image'], sample_batched['label']
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
                        
            # Evaluate the network (forward pass)
            model.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            # Compute the gradient
            loss.backward()
            print(loss)
            # Update the parameters of the model with a gradient step
            optimizer.step()
              
            # Test the quality on the test set
            model.eval()
            accuracies_test = []
            
        model.eval()
        for sample_batched_test in dataloader_test:
            batch_x_test, batch_y_test = sample_batched_test['image'], sample_batched_test['label'] 
            batch_x_test, batch_y_test = batch_x.to(device), batch_y.to(device)
            
            # Evaluate the network (forward pass)
            prediction = model(batch_x_test)
            accuracies_test.append(criterion(prediction, batch_y_test))
        
        print("Epoch {} | Test IoU: {:.5f}".format(epoch, sum(accuracies_test).item()/len(accuracies_test)))


def main(num_epochs = 10, learning_rate = 1e-3, batch_size = 4, train_percentage = 0.8):  
    # If a GPU is available (should be on Colab, we will use it)
    """
    if not torch.cuda.is_available():
        raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    roof_dataset = AvailableRooftopDataset(dir_images = '../data/images/', dir_labels = '../data/labels/')
    dataset_length = len(roof_dataset)
    train_dataset_length = int(dataset_length*train_percentage)
    test_dataset_length = dataset_length - train_dataset_length
    roof_dataset_train, roof_dataset_test = torch.utils.data.random_split(roof_dataset, [train_dataset_length, test_dataset_length],
                                                                          generator=torch.Generator().manual_seed(42))
    
    roof_dataloader_train = DataLoader(roof_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    roof_dataloader_test = DataLoader(roof_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    #criterion = IOULoss()
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = GeneralLoss(jaccard_distance_loss)
    criterion = DiceLoss()
    
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, criterion, roof_dataloader_train, roof_dataloader_test, optimizer, num_epochs, device)
    

if __name__ == "__main__":
    main()
