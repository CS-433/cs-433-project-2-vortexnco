# CS-433 Machine Learning project 2

## Automatic detection of available area for rooftop solar panel installations

*by Ghali Chraibi, Auguste Baum, Alexander Apostolov*

In this project, we provide Python code to detect rooftop available surface for installing PV modules in aerial images using Machine Learning. The model we use is a U-Net.


### Repository structure

- `src/` contains most of the code for this project:
    - `model/` contains the code for the PyTorch implementation of the U-Net that has been adapted from [this repository](https://github.com/milesial/Pytorch-UNet).
    - `AvailableRooftopDataset.py` code to create the dataset of PV and noPV images and their respective labels.
    - `load_data.py` code to create the train, validation and test dataloaders with the repective transformations for the train dataloader.
    - `post_processing.py` code to select the the threshold for decisions after the model has been trained and to evaluate on various metrics on the validation or test set.
    - `training.py` code to train the U-Net.
    - `visualisation.py` code to vizualize the predictions of the model.
    - `dataset_vizualisation.ipynb` Jupyter Notebook that shows images and labels with their transforms & show the computation of the weights for the w-BCE loss.
    - `training_example.ipynb` Jupyter Notebook that shows how the best model is trained.
    - `post_processing_example.ipynb` Jupyter Notebook that shows how the threshold for decisions is chosen on the validation set and gives examples of some predictions and standard metrics on the test set.

    The Jupyter Notebooks are further explained in [the following section](#how-to-reproduce-the-results).
- `labelling_tool/` contains the adapted labelling tool based on OpenCV to create the labels.
- `report/` contains various files to cret the report for this project.
- `requirements.txt` all the external libraries needed to run the code.
- `data/` contains the data for this project:
    - `train/` training data, randomly chosen 70% of the data
        - `PV/` .png PV images.
        - `noPV/` .png noPV images.
        - `labels/` .png labels of PV images.
    - `validation/` validation data, randomly chosen 15% of the data
        - `PV/` .png PV images.
        - `noPV/` .png noPV images.
        - `labels/` .png labels of PV images.
    - `test/` test data, randomly chosen 15% of the data
        - `PV/` .png PV images.
        - `noPV/` .png noPV images.
        - `labels/` .png labels of PV images.
- `saved_models/` contains the saved_model parameters.

### How to reproduce the results?

All results displayed in the report can be obtained by running the various Jupyter Notebooks. In each of them, the user only has to run the cells, random seeds are already set. The data can be obtained [here](https://drive.switch.ch/index.php/s/rLs7MnYr5mgp2cS) and the model parameters of our best model can be obtained [here](https://drive.switch.ch/index.php/s/21F8TGvd8UbCPJc).

- `dataset_vizualization.ipynb`<br/>
This Jupyter Notebook shows examples of the data with their transforms (for the train set). The computation of the weights used for the w-BCE is also shown.

- `training_example.ipynb`<br/>
This Jupyter Notebook trains a model with specific hyperparameters that can be set up by the user. The default values in the notebook will train the best model presented in the report. Running this notebook locally without a GPU takes significant time, so a notebook that has already been run is available on Google Colab. After training the model, its weights will be saved in the `saved_models` folder.

- `post_processing_example.ipynb`<br/>
This Jupyter Notebook shows:
  - How we choose the threshold over which we consider that the model predicts that the pixel is available, and
  - How we evaluate a model once the "best" threshold has been decided on.
By default it will do this for our best model (described in the report). A user can repeat these steps for any other model that has been trained and whose weights have been saved.
