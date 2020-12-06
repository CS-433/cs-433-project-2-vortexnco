import torch

from AvailableRooftopDataset import AvailableRooftopDataset
from torch.utils.data import DataLoader
from model.unet_model import UNet
from losses import GeneralLoss, jaccard_loss, jaccard_distance_loss, DiceLoss
from torchvision import transforms
import os


def train(model, criterion, dataloader_train, dataloader_test, optimizer, num_epochs, device, saving_frequency = 2):
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
    
    print("Starting training during {} epochs".format(num_epochs))
    avg_train_error = []
    avg_test_error = []

    for epoch in range(num_epochs):
        if (epoch + 1 % saving_frequency == 0):
            with open("errors.txt", "w") as f:
                f.write("Epoch {}".format(epoch))
                f.write(str(avg_train_error))
                f.write(str(avg_test_error))
        model.train()

        train_error = []
        for batch_x, batch_y in dataloader_train:
            # batch_x, batch_y = sample_batched['image'], sample_batched['label']
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)

            # Evaluate the network (forward pass)
            model.zero_grad()
            output = model(batch_x)
            
            #output is Bx1xHxW and batch_y is BxHxW, squeezing first dimension of output to have same dimension
            loss = criterion(torch.squeeze(output, 1), batch_y)
            train_error.append(loss)

            # Compute the gradient
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        # Test the quality on the whole training set
        avg_train_error.append(sum(train_error).item() / len(train_error))
        
        # Test the quality on the test set
        model.eval()
        accuracies_test = []

        for batch_x_test, batch_y_test in dataloader_test:
            # batch_x_test, batch_y_test = sample_batched_test['image'], sample_batched_test['label']
            batch_x_test, batch_y_test = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x_test)
            accuracies_test.append(criterion(prediction, batch_y_test))
        avg_test_error.append(sum(accuracies_test).item() / len(accuracies_test))

        print( "Epoch {} | Train Error: {:.5f}, Test Error: {:.5f}".format( epoch, avg_train_error[-1], avg_test_error[-1] ))
    
    #Writing final results on the file
    with open("errors.txt", "w") as f:
        f.write("Epoch {}".format(epoch))
        f.write(str(avg_train_error))
        f.write(str(avg_test_error))

    return avg_train_error, avg_test_error


def main(num_epochs=10, learning_rate=1e-3, batch_size=4, train_percentage=0.8, dir_data = "/raid/machinelearning_course/data/", saving_frequency=2):
    """
    if not torch.cuda.is_available():
        raise Exception("Things will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    

    # Instantiate the dataset
    roof_dataset = AvailableRooftopDataset(
        dir_images = os.path.join(dir_data, "images"),#dir_data + "images/",
        dir_labels = os.path.join(dir_data, "labels"), #dir_data + "labels/",
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(250, ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )

    # Split the dataset in train_set and test_set
    dataset_length = len(roof_dataset)
    train_dataset_length = int(dataset_length * train_percentage)
    test_dataset_length = dataset_length - train_dataset_length
    roof_dataset_train, roof_dataset_test = torch.utils.data.random_split(
        roof_dataset,
        [train_dataset_length, test_dataset_length],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders associated to train/test set
    roof_dataloader_train = DataLoader(
        roof_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    roof_dataloader_test = DataLoader(
        roof_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # criterion = IOULoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = GeneralLoss(jaccard_distance_loss)
    weight_for_positive_class = 5.
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_for_positive_class]))
    #criterion = DiceLoss()

    # To load model params from a file
    # model = TheModelClass(*args, **kwargs)
    # For us: model = UNet(n_channels=3, n_classes=1, bilinear=False)
    # model.load_state_dict(torch.load(PATH))
    # model.eval() 

    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    avg_train_error, avg_test_error = train(
        model,
        criterion,
        roof_dataloader_train,
        roof_dataloader_test,
        optimizer,
        num_epochs,
        device,
        saving_frequency
    )

    # To save model params to a file
    # torch.save(model.state_dict(), PATH)

    print(avg_train_error, avg_test_error)


if __name__ == "__main__":
    main(num_epochs=300, batch_size=50)
