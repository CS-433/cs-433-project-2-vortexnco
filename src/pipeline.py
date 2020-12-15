import torch

from AvailableRooftopDataset import AvailableRooftopDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from model.unet_model import UNet
from torchvision import transforms
import os


def load_data(
    dir_data_training: str,
    dir_data_validation: str,
    dir_data_test: str,
    prop_noPV_training: float,
    min_rescale_images: float,
    batch_size: int
):
    """
    Create the DataLoader objects that will generate the training, validation and test sets.

    Parameters
    ----------
    dir_data_training : str
        Directory where the folders "/images", "/labels" and "noPV/" are for the training set.
    dir_data_validation : str
        Directory where the folders "/images", "/labels" and "noPV/" are for the validation set.
    dir_data_test : str
        Directory where the folders "/images", "/labels" and "noPV/" are for the test set.
    prop_noPV_training : float
        Proportion of noPV images to add for the training of the model.
    min_rescale_images : float
        Minimum proportion of the image to keep for the RandomResizedCrop transform.
    batch_size : int
        Number of samples per batch in the DataLoaders.

    Returns
    -------
    train_dl : torch.utils.data.DataLoader
        Training DataLoader.
    validation_dl : torch.utils.data.DataLoader
        Validation DataLoader.
    test_dl : torch.utils.data.DataLoader
        Test DataLoader.
    """

    ## Define transforms
    # Transforms to augment the data (for training set)
    transform_aug = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                250, scale=(min_rescale_images, 1.0), ratio=(1.0, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # Identity transforms (for validation and test set)
    transform_id = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )

    ## Define datasets
    # Instantiate the training dataset
    roof_train_dataset = AvailableRooftopDataset(
        dir_PV=os.path.join(dir_data_training, "PV"),
        dir_noPV=os.path.join(dir_data_training, "noPV"),
        dir_labels=os.path.join(dir_data_training, "labels"),
        transform=transform_aug,
        prop_noPV=prop_noPV_training,
    )

    # Instantiate the validation dataset
    roof_validation_dataset = AvailableRooftopDataset(
        dir_PV=os.path.join(dir_data_validation, "PV"),
        dir_noPV=os.path.join(dir_data_validation, "noPV"),
        dir_labels=os.path.join(dir_data_validation, "labels"),
        transform=transform_id,
        prop_noPV=1.0,
    )

    # Instantiate the test dataset
    roof_test_dataset = AvailableRooftopDataset(
        dir_PV=os.path.join(dir_data_test, "PV"),
        dir_noPV=os.path.join(dir_data_test, "noPV"),
        dir_labels=os.path.join(dir_data_test, "labels"),
        transform=transform_id,
        prop_noPV=1.0,
    )

    ## Instantiate DataLoader objects for each dataset (train/validation/test)
    roof_train_dl = DataLoader(
        roof_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    roof_validation_dl = DataLoader(
        roof_validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    roof_test_dl = DataLoader(
        roof_test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return roof_train_dl, roof_validation_dl, roof_test_dl


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
    optimizer_type : str = "ADAM",
    loss : str = "BCE",
    use_scheduler: bool = False,
    milestones_scheduler: list = None,
    gamma_scheduler: float = None,
    batch_size: int = 32,
    # dir_data: str = "/raid/machinelearning_course/data/",
    dir_data_training: str = "../data/train",
    dir_data_validation: str = "../data/validation",
    dir_data_test: str = "../data/test",
    prop_noPV_training: float = 0.0,
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
    optimizer_type : str, optional
        Can be "ADAM" or "SGD". The default is "ADAM".
    loss : str, optional
        Cane be "BCE" of "L1". The default is "BCE".
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
    dir_data_training : str, optional
        Directory where the folders "/images", "/labels" and "noPV/" are for the training set.
    dir_data_validation : str, optional
        Directory where the folders "/images", "/labels" and "noPV/" are for the validation set.
    dir_data_test : str, optional
        Directory where the folders "/images", "/labels" and "noPV/" are for the test set.
    prop_noPV_training : float, optional
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
        dir_data_training,
        prop_noPV_training,
        min_rescale_images,
        batch_size
    )

    if loss == "BCE":
        # Create Binary cross entropy loss weighted according to positive pixels.
        # pos_weight > 1 increases recall.
        # pos_weight < 1 increases precision.
        pos_weight = torch.tensor([weight_for_positive_class]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss == "L1":
        criterion = torch.nn.L1Loss()
    else : 
        raise NotImplementedError(f"{loss} is not implemented.")

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
        if optimizer_type == "ADAM":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else : 
            raise NotImplementedError(f"{optimizer} is not implemented.")
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
