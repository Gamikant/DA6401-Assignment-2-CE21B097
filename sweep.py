import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import json
import argparse
import os
from torchvision import transforms
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

def train_model(config=None):
    # Initialize wandb run
    with wandb.init(config=config, project=args.project, entity=args.entity) as run:
        # Access all hyperparameter values from wandb.config
        config = wandb.config
        
        # Create data loaders with or without augmentation
        if config.data_augmentation == 'Yes':
            # Apply data augmentation to training data
            train_transform = get_data_augmentation_transform(img_size=config.img_size)
            train_dataset = datasets.ImageFolder(root=f"{args.data_dir}/train", transform=train_transform)
            
            # Get targets for stratified split
            targets = np.array(train_dataset.targets)
            
            # Perform stratified split
            train_indices, val_indices = train_test_split(
                np.arange(len(targets)),
                test_size=0.2,
                shuffle=True,
                stratify=targets,
                random_state=42
            )
            
            # Create subset datasets
            train_subset = Subset(train_dataset, train_indices)
            
            # For validation, use standard transform without augmentation
            val_transform = transforms.Compose([
                transforms.Resize((config.img_size, config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            val_dataset = datasets.ImageFolder(root=f"{args.data_dir}/train", transform=val_transform)
            val_subset = Subset(val_dataset, val_indices)
        else:
            # Use standard transforms without augmentation
            transform = transforms.Compose([
                transforms.Resize((config.img_size, config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            train_dataset = datasets.ImageFolder(root=f"{args.data_dir}/train", transform=transform)
            
            # Get targets for stratified split
            targets = np.array(train_dataset.targets)
            
            # Perform stratified split
            train_indices, val_indices = train_test_split(
                np.arange(len(targets)),
                test_size=0.2,
                shuffle=True,
                stratify=targets,
                random_state=42
            )
            
            # Create subset datasets
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=config.batch_size, shuffle=True, num_workers=2
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=config.batch_size, shuffle=False, num_workers=2
        )
        
        # Get number of classes
        num_classes = len(train_dataset.classes)
        
        # Create model with the specified hyperparameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FlexibleCNN(
            input_channels=3,
            num_classes=num_classes,
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
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        if config.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:  # SGD
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        
        # Variables to track best model
        best_val_acc = 0.0
        best_model_path = os.path.join(wandb.run.dir, "best_model.pth")
        
        # Train the model
        for epoch in range(config.epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            print(f'Epoch {epoch+1}/{config.epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                
                # Save the hyperparameters as a JSON file
                config_dict = {k: v for k, v in config.items()}
                config_dict['best_val_acc'] = best_val_acc
                config_dict['run_id'] = run.id
                
                with open('best_params.json', 'w') as f:
                    json.dump(config_dict, f, indent=4)
                
                # Also save as an artifact
                artifact = wandb.Artifact(
                    name=f"best-model-{run.id}", 
                    type="model",
                    description=f"Best model with val_acc: {best_val_acc:.4f}"
                )
                artifact.add_file(best_model_path)
                artifact.add_file('best_params.json')
                run.log_artifact(artifact)
        
        # At the end of training, also save the final model
        final_model_path = os.path.join(wandb.run.dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        
        # Log the final model as an artifact
        artifact = wandb.Artifact(
            name=f"final-model-{run.id}", 
            type="model",
            description=f"Final model with val_acc: {val_acc:.4f}"
        )
        artifact.add_file(final_model_path)
        run.log_artifact(artifact)

def main():
    # Import needed modules
    import json
    
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'num_filters': {
                'values': [10]
            },
            'filter_size': {
                'values': [5]
            },
            'activation': {
                'values': ['ReLU'] #, 'GELU', 'SiLU', 'Mish']
            },
            'filter_org': {
                'values': ['same'] #, 'double', 'half']
            },
            'batch_norm': {
                'values': ['Yes'] #, 'No']
            },
            'dropout': {
                'values': [0.1] #, 0.0, 0.2]
            },
            'data_augmentation': {
                'values': ['No']
            },
            'dense_neurons': {
                'values': [32]
            },
            'batch_size': {
                'values': [256]
            },
            'learning_rate': {
                'values': [0.001]
            },
            'optimizer': {
                'values': ['Adam'] #, 'SGD']
            },
            'epochs': {
                'value': 10
            },
            'img_size': {
                'value': 224
            }
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    
    # Run the sweep
    wandb.agent(sweep_id, train_model, count=args.num_runs)
    
    # After all runs are complete, find the best run
    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", {"sweep": sweep_id})
    
    best_run = None
    best_val_acc = 0.0
    
    for run in runs:
        if run.summary.get('val_acc', 0) > best_val_acc:
            best_val_acc = run.summary.get('val_acc', 0)
            best_run = run
    
    if best_run:
        print(f"Best run: {best_run.name} with val_acc: {best_val_acc:.4f}")
        
        # Get the best model artifact
        artifacts = best_run.logged_artifacts()
        best_model_artifact = None
        
        for artifact in artifacts:
            if artifact.type == "model" and "best-model" in artifact.name:
                best_model_artifact = artifact
                break
        
        if best_model_artifact:
            # Download the artifact
            artifact_dir = best_model_artifact.download()
            
            # Copy the model and parameters to a standard location
            import shutil
            shutil.copy(os.path.join(artifact_dir, "best_model.pth"), "best_model.pth")
            shutil.copy(os.path.join(artifact_dir, "best_params.json"), "best_params.json")
            
            print(f"Best model and parameters saved to current directory")
        else:
            print("Could not find best model artifact")
    else:
        print("Could not find best run")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for CNN model")
    parser.add_argument("--data_dir", type=str, default="inaturalist_12k", help="Path to dataset directory")
    parser.add_argument("--project", type=str, default="da6401-Assignment-2-CE21B097", help="WandB project name")
    parser.add_argument("--entity", type=str, default="ce21b097-indian-institute-of-technology-madras", help="WandB entity name")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs to perform in the sweep")
    
    args = parser.parse_args()
    
    # Import needed modules inside main to avoid circular imports
    import numpy as np
    from torchvision import datasets
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split
    
    main()
