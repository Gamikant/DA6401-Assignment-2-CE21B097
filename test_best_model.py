import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from model import FlexibleCNN
from dataset_split import create_stratified_split

# Dictionary to map activation function names to PyTorch classes
ACTIVATION_FUNCTIONS = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'Mish': nn.Mish,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU
}

def get_data_augmentation_transform(img_size=224):
    """Create a transform with data augmentation"""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transform(img_size=224):
    """Create a transform for test data"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_prediction_grid(images, labels, predictions, class_names, confidences=None):
    """Create a creative 10x3 grid of images with their true labels and predictions"""
    plt.figure(figsize=(15, 30))
    
    # Define class icons or emojis for visual enhancement
    class_icons = {
        'Amphibia': '🐸',
        'Animalia': '🦓',
        'Arachnida': '🕷️',
        'Aves': '🦅',
        'Fungi': '🍄',
        'Insecta': '🐝',
        'Mammalia': '🦁',
        'Mollusca': '🐚',
        'Plantae': '🌿',
        'Reptilia': '🦎'
    }
    
    # Create a 10x3 grid
    for i in range(min(30, len(images))):
        # Get the image, true label, and prediction
        img = images[i]
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        
        # Denormalize the image
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy for matplotlib
        img = img.permute(1, 2, 0).numpy()
        
        # Plot the image
        plt.subplot(10, 3, i + 1)
        plt.imshow(img)
        
        # Set title color based on prediction correctness
        is_correct = true_label == pred_label
        title_color = 'green' if is_correct else 'red'
        
        # Get icons for true and predicted classes
        true_icon = class_icons.get(true_label, '')
        pred_icon = class_icons.get(pred_label, '')
        
        # Create title with icons
        title = f"True: {true_icon} {true_label}\nPred: {pred_icon} {pred_label}"
        
        # Add confidence score if available
        if confidences is not None:
            title += f"\nConf: {confidences[i]:.2f}"
        
        # Add a checkmark or X to indicate correctness
        correctness_symbol = "✓" if is_correct else "✗"
        title += f" {correctness_symbol}"
        
        plt.title(title, color=title_color)
        plt.axis('off')
        
        # Add a colored border based on correctness
        plt.gca().spines['top'].set_color(title_color)
        plt.gca().spines['bottom'].set_color(title_color)
        plt.gca().spines['left'].set_color(title_color)
        plt.gca().spines['right'].set_color(title_color)
        plt.gca().spines['top'].set_linewidth(5)
        plt.gca().spines['bottom'].set_linewidth(5)
        plt.gca().spines['left'].set_linewidth(5)
        plt.gca().spines['right'].set_linewidth(5)
    
    plt.tight_layout()
    plt.savefig('prediction_grid.png', dpi=300, bbox_inches='tight')
    print("Prediction grid saved as 'prediction_grid.png'")

def main():
    # Check if best_params.json exists
    if os.path.exists('best_params.json'):
        # Load the best parameters
        with open('best_params.json', 'r') as f:
            best_config_dict = json.load(f)
        
        # Convert the dictionary to an argparse.Namespace object
        config = argparse.Namespace(**best_config_dict)
        
        print(f"Loaded best parameters from best_params.json:")
        for key, value in best_config_dict.items():
            print(f"  {key}: {value}")
        
        # Create data loaders
        print("Loading datasets...")
        test_transform = get_test_transform(img_size=config.img_size)
        
        # Load datasets
        test_dataset = datasets.ImageFolder(root=f"{args.data_dir}/val", transform=test_transform)
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        
        # Get class names
        class_names = test_dataset.classes
        print(f"Classes: {class_names}")
        
        # Create model with the best hyperparameters
        print("Creating model with the best hyperparameters...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FlexibleCNN(
            input_channels=3,
            num_classes=len(class_names),
            num_filters=config.num_filters,
            filter_size=config.filter_size,
            activation_fn=ACTIVATION_FUNCTIONS[config.activation],
            dense_neurons=config.dense_neurons,
            input_size=config.img_size,
            filter_org=config.filter_org,
            use_batchnorm=(config.batch_norm == 'Yes'),
            dropout_rate=config.dropout
        )
        model = model.to(device)
        
        # Load the saved model weights
        if os.path.exists('best_model.pth'):
            print("Loading saved model weights...")
            model.load_state_dict(torch.load('best_model.pth', map_location=device))
        else:
            print("Error: best_model.pth not found. Cannot load model weights.")
            return
        
        # Evaluate on test data
        print("Evaluating on test data...")
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # For visualization
        all_images = []
        all_labels = []
        all_preds = []
        all_confidences = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                
                # Get predictions and confidences
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = probabilities.max(1)
                
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                # Store some images, labels, predictions, and confidences for visualization
                if len(all_images) < 30:  # Store 30 images for the 10x3 grid
                    batch_size = inputs.size(0)
                    num_to_add = min(batch_size, 30 - len(all_images))
                    all_images.extend(inputs[:num_to_add].cpu())
                    all_labels.extend(labels[:num_to_add].cpu())
                    all_preds.extend(predicted[:num_to_add].cpu())
                    all_confidences.extend(confidences[:num_to_add].cpu())
        
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        
        # Create and save the visualization grid
        create_prediction_grid(all_images[:30], all_labels[:30], all_preds[:30], class_names, all_confidences[:30])
    else:
        print("Error: best_params.json not found. Please run sweep.py first to generate best parameters.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the best model on test data")
    parser.add_argument("--data_dir", type=str, default="inaturalist_12k", help="Path to dataset directory")
    
    args = parser.parse_args()
    main()