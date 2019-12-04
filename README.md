# PyTorch Project Template

## Guidelines

- Take care of the matrix dimensions, axis and datatypes (Pytorch typecast is different from Numpy). 
- Take care of variable referencing (whether its by value or by address).
- Use assert() and keepdim options.
- List all non-trivial options of any function even if they take default values for clarity. 
- Use space after comma in function parameters for clarity but not in tuples
- Add shell scripts or links in README for downloading the pretrained weights and the datasets. Add these large folders to `.gitignore` to avoid uploading them.

## Folder Structure

#### Folders
- `checkpoints` - Folder to store the partially trained models and intermediate plots while training.
- `data` - Folder which contains files related to the dataset, specifically the custom dataset classes and other necessary utilities.
- `demo` - Folder to add the data samples on which we would like to run a demo.
- `final` - Folder to store final trained models and plots.
- `models` - Folder which contains model definition files.
- `utils` - Folder for utility functions - loss functions, metrics, decoders and other general functions.

#### Files
- `checker.py` - A file with checker functions for testing all the modules and functions in the project as well as any other checks to be performed.
- `config.py` - Configuration options and hyperparameter values.
- `demo.py` - Python script for running the trained model on the samples in the `demo` folder. 
- `test.py` - Python script to test the trained model on the test set and obtain the loss value and evaluation metric value on the test set.
- `train.py` - Python script to train the models.

## Changes Required


