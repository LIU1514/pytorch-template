# PyTorch Project Template

## Project Structure

```
pytorch-template/
│
│
├── data/ - custom dataset class definitions and data-related functions
│   ├── my_dataset.py
│   └── tools.py
│
├── models/ - model class definitions
│   └── my_net.py
│
├── essentials/
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
├── inspect.py - functions to test modules and perform other checks
│
├── preprocess.py - preprocess all data samples
│
├── train.py - trains the model on the train set
│
├── test.py - tests the model on the test set
│
└── demo.py - generates model predictions for samples
```
