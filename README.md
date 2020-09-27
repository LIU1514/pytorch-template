# PyTorch Project Template

## Project Structure

```
pytorch-template/
│
│
├── data/ - custom dataset class definitions and data-related functions
│   ├── __init__.py
│   ├── my_dataset.py
│   └── tools.py
│
├── models/ - model class definitions
│   ├── __init__.py
│   └── my_net.py
│
├── essentials/
│   ├── __init__.py
│   ├── pprocs.py - preprocessing functions
│   ├── core.py - train and evaluate the models
│   ├── losses.py - custom loss class definitions
│   ├── decoders.py - decoder functions
│   └── metrics.py - functions to compute evaluation metrics
│
│
├── checkpoints/ - stores intermediate model weights, loss/metric plots
│   ├── plots/
│   └── weights/
│
├── saved/ - stores final trained model weights, loss/metric plots
│   ├── plots/
│   └── weights/
│
│
├── config.py - configuration options and hyperparameter values
│
├── checks.py - functions to test modules and perform other checks
│
├── preprocess.py - preprocess all data samples
│
├── train.py - trains the model on the train set
│
├── test.py - tests the model on the test set
│
└── demo.py - generates model predictions for samples
```
