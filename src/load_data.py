import os

from torchvision import transforms
from AvailableRooftopDataset import AvailableRooftopDataset
from torch.utils.data import DataLoader



def load_data(
    prop_noPV_training: float,
    min_rescale_images: float,
    batch_size: int,
    dir_data_training: str = "",
    dir_data_validation: str = "",
    dir_data_test: str = "",
):
    """
    Create the DataLoader objects that will generate the training, validation and test sets.

    Parameters
    ----------
    dir_data_training : str
        Directory where the folders "images/", "labels/" and "noPV/" are for the training set.
        If empty, the data is not generated.
    dir_data_validation : str
        Directory where the folders "images/", "labels/" and "noPV/" are for the validation set.
        If empty, the data is not generated.
    dir_data_test : str
        Directory where the folders "images/", "labels/" and "noPV/" are for the test set.
        If empty, the data is not generated.
    prop_noPV_training : float
        Proportion of noPV images to add for the training of the model.
    min_rescale_images : float
        Minimum proportion of the image to keep for the RandomResizedCrop transform.
    batch_size : int
        Number of samples per batch in the DataLoaders.

    Returns
    -------
    train_dl : torch.utils.data.DataLoader
        Training DataLoader, if data directory is provided, otherwise None.
    validation_dl : torch.utils.data.DataLoader
        Validation DataLoader, if data directory is provided, otherwise None.
    test_dl : torch.utils.data.DataLoader
        Test DataLoader, if data directory is provided, otherwise None.
    """
    roof_train_dataset = None
    if dir_data_training:
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
        # Instantiate the training dataset
        roof_train_dataset = AvailableRooftopDataset(
            dir_PV=os.path.join(dir_data_training, "PV"),
            dir_noPV=os.path.join(dir_data_training, "noPV"),
            dir_labels=os.path.join(dir_data_training, "labels"),
            transform=transform_aug,
            prop_noPV=prop_noPV_training,
        )

    # No transform applied to validation and train images (the model should not need
    # any preprocessing)
    transform_id = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    # Instantiate the validation and test datasets
    roof_validation_dataset, roof_test_dataset = (
        AvailableRooftopDataset(
            dir_PV=os.path.join(dir_data, "PV"),
            dir_noPV=os.path.join(dir_data, "noPV"),
            dir_labels=os.path.join(dir_data, "labels"),
            transform=transform_id,
            prop_noPV=-1, # All of them
        )
        if dir_data
        else None
        for dir_data in (dir_data_validation, dir_data_test)
    )

    # Instantiate the DataLoaders
    roof_train_dl, roof_validation_dl, roof_test_dl = (
        DataLoader(roof_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        if roof_dataset
        else None
        for roof_dataset in (
            roof_train_dataset,
            roof_validation_dataset,
            roof_test_dataset,
        )
    )

    return roof_train_dl, roof_validation_dl, roof_test_dl