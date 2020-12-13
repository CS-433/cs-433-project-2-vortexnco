import torch

from AvailableRooftopDataset import AvailableRooftopDataset
from torch.utils.data import DataLoader
from helpers import get_DataLoaders
from torch.optim.lr_scheduler import MultiStepLR
from model.unet_model import UNet
from torchvision import transforms
import os


def load_data(
    dir_data: str,
    prop_noPV: float,
    min_rescale_images: float,
    batch_size: int,
    train_percentage: float,
    validation_percentage: float,
):
    """
    Create the DataLoader objects that will generate the training, validation and test sets.

    Parameters
    ----------
    dir_data : str
        Directory where the folders "/images", "/labels" and "noPV/" are.
    prop_noPV : float
        Proportion of all noPV images to add.
    min_rescale_images : float
        Minimum proportion of the image to keep for the RandomResizedCrop transform.
    batch_size : int
        Number of samples per batch in the DataLoaders.
    train_percentage : float
        Percentage of the Dataset to be used for Training.
    validation_percentage : float
        Percentage of the Dataset to be used for Validation.

    Returns
    -------
    train_set : torch.utils.data.DataLoader
        Training DataLoader.
    validation_set : torch.utils.data.DataLoader
        Validation DataLoader.
    test_set : torch.utils.data.DataLoader
        Test DataLoader.
    """

    # Choose which transforms to apply
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                250, scale=(min_rescale_images, 1.0), ratio=(1.0, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # Instantiate Dataset object
    roof_dataset = AvailableRooftopDataset(
        dir_PV=os.path.join(dir_data, "PV"),
        dir_noPV=os.path.join(dir_data, "noPV"),
        dir_labels=os.path.join(dir_data, "labels"),
        transform=transform,
        prop_noPV=prop_noPV,
    )

    # Instantiate DataLoader objects according to the splits
    test_percentage = 1 - train_percentage - validation_percentage
    train_set, validation_set, test_set = get_DataLoaders(
        roof_dataset,
        train_percentage,
        validation_percentage,
        test_percentage,
        batch_size,
    )

    return train_set, validation_set, test_set


def train(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_validation: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    use_scheduler: bool,
    scheduler: torch.optim.lr_scheduler,
    num_epochs: int,
    device,
    file_losses: str,
    saving_frequency: int,
):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    criterion : torch.nn.modules.loss._Loss
        Criterion (Loss) to use during training.
    dataloader_train : torch.utils.data.DataLoader
        Dataloader for training.
    dataloader_validation : torch.utils.data.DataLoader
        Dataloader to validate the model during training after each epoch.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    use_scheduler : bool
        If True, uses a  MultiStepLR scheduler to adapt the learning rate during training.
    scheduler : torch.optim.lr_scheduler.MultiStepLR
        Scheduler to use to adapt learning rate during training.
    num_epochs : int
        Number of epochs to train for.
    device :
        Device on which to train (GPU or CPU cuda devices)
    file_losses : str
        Name of the file in which to save the Train and Test losses.
    saving_frequency : int
        Frequency at which to save Train and Test loss on file.

    Returns
    -------
    avg_train_error, avg_validation_error : list of float, list of float
        List of Train errors or losses after each epoch.
        List of Validation errors or losses after each epoch.

    """

    print("Starting training during {} epochs".format(num_epochs))
    avg_train_error = []
    avg_validation_error = []

    for epoch in range(num_epochs):

        # Writing results to file regularly in case of interruption during training.
        if epoch + 1 % saving_frequency == 0:
            with open(file_losses, "w") as f:
                f.write("Epoch {}".format(epoch))
                f.write(str(avg_train_error))
                f.write(str(avg_validation_error))

        model.train()
        train_error = []
        for batch_x, batch_y in dataloader_train:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(
                device, dtype=torch.float32
            )

            # Evaluate the network (forward pass)
            model.zero_grad()
            output = model(batch_x)

            # output is Bx1xHxW and batch_y is BxHxW, squeezing first dimension of output to have same dimension
            loss = criterion(torch.squeeze(output, 1), batch_y)
            train_error.append(loss)

            # Compute the gradient
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        # Each scheduler step is done after a hole epoch
        # Once milestones epochs are reached the learning rates is decreased.
        if use_scheduler:
            scheduler.step()

        # Test the quality on the whole training set (overestimating the true value)
        avg_train_error.append(sum(train_error).item() / len(train_error))

        # Validate the quality on the validation set
        model.eval()
        accuracies_validation = []
        with torch.no_grad():
            for batch_x_validation, batch_y_validation in dataloader_validation:
                batch_x_validation, batch_y_validation = (
                    batch_x_validation.to(device, dtype=torch.float32),
                    batch_y_validation.to(device, dtype=torch.float32),
                )
                # Evaluate the network (forward pass)
                prediction = model(batch_x_validation)
                accuracies_validation.append(
                    criterion(torch.squeeze(prediction, 1), batch_y_validation)
                )
            avg_validation_error.append(
                sum(accuracies_validation).item() / len(accuracies_validation)
            )

        print(
            "Epoch {} | Train Error: {:.5f}, Validation Error: {:.5f}".format(
                epoch, avg_train_error[-1], avg_validation_error[-1]
            )
        )

    # Writing final results on the file
    with open(file_losses, "w") as f:
        f.write("Epoch {}".format(epoch))
        f.write(str(avg_train_error))
        f.write(str(avg_validation_error))

    return avg_train_error, avg_validation_error


def main(
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    use_scheduler: bool = False,
    milestones_scheduler: list = None,
    gamma_scheduler: float = None,
    batch_size: int = 32,
    train_percentage: float = 0.7,
    validation_percentage: float = 0.15,
    # test_percentage: float = 0.15,
    # dir_data: str = "/raid/machinelearning_course/data/",
    dir_data: str = "../data/",
    prop_noPV: float = 0.0,
    min_rescale_images: float = 0.6,
    file_losses: str = "losses.txt",
    saving_frequency: int = 2,
    weight_for_positive_class: float = 5.25,
    save_model_parameters: bool = False,
    load_model_parameters: bool = False,
    dir_for_model_parameters: str = "../saved_models",
    filename_model_parameters_to_load: str = None,
    filename_model_parameters_to_save: str = None,
):
    """
    Main training function with tunable parameters.

    Parameters
    ----------
    num_epochs : int, optional
        Number of epochs to train. The default is 100.
    learning_rate : float, optional
        Learning rate of the Optimizer. The default is 1e-3.
    use_scheduler : bool
        If True, use a MultiStepLR. You should the next two parameters if used.
    milestones_scheduler : list
        List of epochs at which to adapt the learning rate.
    gamma_scheduler : float
        Value by which to multiply the learning rate at each of the previously
        define milestone epochs.
        Example values are 0.5 or 0.1.
    batch_size : int, optional
        Number of samples per batch in the Dataloaders. The default is 32.
    train_percentage : float, optional
        Percentage of the Dataset to be used for Training. The default is 0.7.
    validation_percentage : float, optional
        Percentage of the Dataset to be used for Validation. The default is 0.15.
    dir_data : str, optional
        Directory where the folders "/images", "/labels" and "noPV/" are.
        The default is "/raid/machinelearning_course/data/".
    prop_noPV : float, optional
        Proportion of all noPV images to add. The default is 0.0.
    min_rescale_images : float, optional
        Minimum proportion of the image to keep for the RandomResizedCrop transform.
        The default is 0.6.
    file_losses : str, optional
        Name of the files where to write the Train and test losses during training.
        The default is "losses.txt".
    saving_frequency : int, optional
        Frequency (in number of epochs) at which to write the train and
        test losses in the file.
        Small frequency is used if high risk that training might
        be interrupted to avoid too much lost data.
        The default is 2.
    weight_for_positive_class : float, optional
        Weight for the positive class in the Binary Cross entropy loss.
        According to the Pytorch documentation it should equal to:
        the number of negative pixels / the number of positive pixels.
        The default is 5.25 (calculated with only PV images).
    save_model_parameters : bool, optional
        If True saves the model at the end of training. The default is False.
    load_model_parameters : bool, optional
        If True loads defined parameters in the model before training.
        The default is False.
    dir_for_model_parameters : str, optional
        Diretory where saved parameters are stored.
        The default is "../saved_models".
    filename_model_parameters_to_load : str, optional
        Filename of the parameters to load before training.
        Should be specified if load_model_parameters is True.
        The default is None.
    filename_model_parameters_to_save : str, optional
        Filename of the parameters to save after training.
        Should be defined is save_model_parameters is True.
        The default is None.

    Returns
    -------
    model : torch.nn.Module
        Model after training.
    avg_train_error : list of float
        List of Train errors or losses after each epoch.
    avg_validation_error : list of float
        List of Validation errors or losses after each epoch.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is {}available.".format("" if torch.cuda.is_available() else "NOT "))

    # Instantiate the dataset
    roof_dataloader_train, roof_dataloader_validation, roof_dataloader_test = load_data(
        dir_data,
        # use_noPV,
        prop_noPV,
        min_rescale_images,
        batch_size,
        train_percentage,
        validation_percentage,
    )

    # Create Binary cross entropy loss weighted according to positive pixels.
    # pos_weight > 1 increases recall.
    # pos_weight < 1 increases precision.
    #pos_weight = torch.tensor([weight_for_positive_class]).to(device)
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = torch.nn.L1Loss()

    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(device)

    # If we're not starting from scratch
    if load_model_parameters:
        path_model_parameters_to_load = os.path.join(
            dir_for_model_parameters, filename_model_parameters_to_load
        )
        model.load_state_dict(torch.load(path_model_parameters_to_load))

    # If we're training or retraining a model
    if (num_epochs > 0):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
        if use_scheduler:
            scheduler = MultiStepLR(
                optimizer, milestones=milestones_scheduler, gamma=gamma_scheduler
            )

        avg_train_error, avg_validation_error = train(
            model,
            criterion,
            roof_dataloader_train,
            roof_dataloader_validation,
            optimizer,
            use_scheduler,
            scheduler,
            num_epochs,
            device,
            file_losses,
            saving_frequency,
        )

        if save_model_parameters:
            path_model_parameters_to_save = os.path.join(
                dir_for_model_parameters, filename_model_parameters_to_save
            )
            torch.save(model.state_dict(), path_model_parameters_to_save)

    # Now find the best threshold
    # precision_recall_curve(y_true, probas_pred, *)
    # roc_curve


    print(avg_train_error, avg_validation_error)

    return model, avg_train_error, avg_validation_error


if __name__ == "__main__":
    main()
