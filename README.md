# Deep Learning Assignment: CNN Models for iNaturalist Classification

This repository contains the implementation of Convolutional Neural Network (CNN) models for classifying images from the iNaturalist dataset. The project is divided into two parts: Part A involves building a custom CNN from scratch, while Part B explores fine-tuning pre-trained models.

## Project Structure

```
.
├── Part A/
│   ├── main.py                # Main script for building and analyzing the custom CNN
│   ├── model.py               # Definition of the FlexibleCNN architecture
│   ├── dataset.py             # Data loading utilities
│   ├── dataset_split.py       # Stratified dataset splitting with subset options
│   ├── utils.py               # Utility functions for model analysis
│   ├── sweep.py               # Hyperparameter tuning with Weights &amp; Biases
│   ├── test_best_model.py     # Evaluation of the best model on test data
│   └── best_params.json       # Best hyperparameters from the sweep
│
├── Part B/
│   ├── Q1.md                  # Addressing input/output adaptations for pre-trained models
│   ├── Q2.md                  # Discussion of fine-tuning strategies
│   ├── Q3.py                  # Implementation of fine-tuning with ResNet50
│   └── README.md              # Detailed comparison of fine-tuning vs. training from scratch
│
└── README.md                  # This file
```


## Part A: Custom CNN Implementation

### Question 1: Building a Custom CNN

A flexible CNN architecture with 5 convolutional layers, each followed by activation and max-pooling, and a final dense layer before the output layer.

### Question 2: Hyperparameter Tuning

Hyperparameter optimization using Weights \& Biases to find the best model configuration.

### Question 3: Analysis of Hyperparameter Impact

Interpretation of the hyperparameter sweep results and their impact on model performance.

### Question 4: Model Evaluation

Evaluation of the best model on the test dataset with visualization of predictions.

## Part B: Fine-tuning Pre-trained Models

### Question 1: Pre-trained Model Adaptation

Addressing the challenges of adapting pre-trained ImageNet models to the iNaturalist dataset.

### Question 2: Fine-tuning Strategies

Discussion of various strategies for fine-tuning large pre-trained models efficiently.

### Question 3: Fine-tuning Implementation

Implementation of a selected fine-tuning strategy and comparison with training from scratch.

## Dataset

The iNaturalist dataset contains images of biological specimens across 10 classes:

- Amphibia
- Animalia
- Arachnida
- Aves
- Fungi
- Insecta
- Mammalia
- Mollusca
- Plantae
- Reptilia


## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- scikit-learn
- wandb (for hyperparameter tuning)


## Instructions to Run

### Part A: Custom CNN

1. **Build and analyze the custom CNN model**:

```
python main.py --num_filters 10 --filter_size 3 --dense_neurons 64
```

2. **Run hyperparameter sweep**:

```
python sweep.py --data_dir inaturalist_12k --num_runs 30 --subset_fraction 0.25
```

This will perform a Bayesian optimization sweep with 30 runs using 25% of the dataset.
3. **Evaluate the best model on test data**:

```
python test_best_model.py --data_dir inaturalist_12k
```

This will load the best model from the sweep and evaluate it on the test set, generating a visualization grid of predictions.

### Part B: Fine-tuning Pre-trained Models

1. **Fine-tune a pre-trained ResNet50 model**:

```
python Q3.py
```

This implements the "Freezing Base Layers" strategy, where early layers are frozen while later layers are fine-tuned.

## Results

### Part A: Custom CNN

- The best model from the hyperparameter sweep achieved a validation accuracy of 15.8%.
- Key hyperparameters: 64 filters, GELU activation, double filter organization, batch normalization, and data augmentation.


### Part B: Fine-tuned ResNet50

- The fine-tuned model achieved a validation accuracy of 81.8% after just 5 epochs.
- This represents a significant improvement over the custom CNN trained from scratch, demonstrating the power of transfer learning.


## Visualization

The `test_best_model.py` script generates a 10×3 grid of sample images from the test set with their true labels and model predictions. The visualization includes:

- Color-coded labels (green for correct, red for incorrect)
- Confidence scores
- Class icons for visual enhancement
- Correctness indicators (✓ or ✗)


## Acknowledgments

- The iNaturalist dataset for providing diverse biological specimen images
- Weights \& Biases for hyperparameter tuning capabilities
- PyTorch and torchvision for deep learning frameworks and pre-trained models