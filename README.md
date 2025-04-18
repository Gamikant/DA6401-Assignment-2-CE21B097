# Part A: Custom CNN Implementation for iNaturalist Classification

This repository contains the implementation of a custom Convolutional Neural Network (CNN) for classifying images from the iNaturalist dataset, which includes 10 biological classes. The project focuses on building a flexible CNN architecture from scratch, optimizing its hyperparameters, and evaluating its performance.

## Project Structure

```
.
├── main.py                # Main script for building and analyzing the custom CNN
├── model.py               # Definition of the FlexibleCNN architecture
├── dataset.py             # Data loading utilities
├── dataset_split.py       # Stratified dataset splitting with subset options
├── utils.py               # Utility functions for model analysis
├── sweep.py               # Hyperparameter tuning with Weights &amp; Biases
├── test_best_model.py     # Evaluation of the best model on test data
├── best_model.pth         # Best model on val data
├── best_params.json       # Best hyperparameters from the sweep
├── prediction_grid.png    # Visualization of model predictions
├── artifacts/             # Best parameters and model of all sweeps saved is separate folders inside artifacts
└── inaturalist_12k/       # Dataset directory
                    ├── train/       # Training data into 10 class folders
                    └── val/         # Validation data into 10 class folders

```


## Implementation Details

### FlexibleCNN Architecture

The core of this project is a custom CNN architecture with:

- 5 convolutional layers, each followed by activation and max-pooling
- Flexible configuration options for filter counts, filter sizes, and activation functions
- Support for different filter organization strategies (constant, doubling, halving)
- Optional batch normalization and dropout for regularization
- A dense layer followed by an output layer with 10 neurons (one for each class)


### Hyperparameter Tuning

The project uses Weights \& Biases (wandb) for hyperparameter optimization, exploring:

- Number of filters: 32, 64
- Filter size: 3
- Activation functions: ReLU, GELU, SiLU, Mish
- Filter organization: same, double, half
- Batch normalization: Yes, No
- Dropout rates: 0.0, 0.1
- Data augmentation: Yes, No
- Dense neurons: 128, 256
- Batch sizes: 64, 128
- Learning rates: 0.001, 0.0005
- Optimizers: Adam, SGD


### Dataset Handling

The implementation includes:

- Support for the iNaturalist dataset with 10 classes
- Stratified dataset splitting to ensure class balance
- Option to use a subset of the data for faster experimentation
- Data augmentation capabilities


## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- scikit-learn
- wandb (for hyperparameter tuning)


## Instructions to Run

### 1. Build and Analyze the Custom CNN Model

```bash
python main.py --num_filters 10 --filter_size 3 --dense_neurons 64
```

This will create a CNN model with the specified parameters and analyze its computational complexity and parameter count.

### 2. Run Hyperparameter Sweep

```bash
python sweep.py --data_dir inaturalist_12k --num_runs 30 --subset_fraction 0.25
```

This will perform a Bayesian optimization sweep with 30 runs using 25% of the dataset to find the optimal hyperparameters.

### 3. Evaluate the Best Model on Test Data

```bash
python test_best_model.py --data_dir inaturalist_12k
```

This will load the best model from the sweep and evaluate it on the test set, generating a visualization grid of predictions.

## Results

The best model from the hyperparameter sweep achieved:

- Validation accuracy: 15.8%
- Best configuration:
    - Number of filters: 64
    - Filter size: 3
    - Activation function: GELU
    - Filter organization: double (increasing filters in each layer)
    - Batch normalization: Yes
    - Dropout: 0
    - Data augmentation: Yes
    - Dense neurons: 256
    - Batch size: 64
    - Learning rate: 0.0005
    - Optimizer: Adam


## Visualization

The `test_best_model.py` script generates a 10×3 grid of sample images from the test set with their true labels and model predictions. The visualization includes:

- Color-coded labels (green for correct, red for incorrect)
- Confidence scores for each prediction
- Class icons for visual enhancement
- Correctness indicators (✓ or ✗)


## Model Analysis

The custom CNN model's computational complexity and parameter count can be calculated using the formulas:

- Total computations: 150,528k²m + 16,660k²m² + 49mn + 10n
- Total parameters: 3k²m + 4k²m² + 49mn + 11n + 5m + 10

Where:

- m = number of filters
- k = filter size
- n = number of neurons in the dense layer
