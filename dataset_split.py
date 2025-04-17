import torch
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

def create_stratified_split(dataset_dir, img_size=224, batch_size=32, val_size=0.2):
    """
    Create a stratified split of the training data, ensuring each class is equally represented
    in the validation set.
    
    Args:
        dataset_dir: Path to the dataset directory containing train and test folders
        img_size: Size to resize images to
        batch_size: Batch size for data loaders
        val_size: Proportion of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Define transforms for training (with optional augmentation) and validation/testing
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the full training dataset
    train_dataset = datasets.ImageFolder(root=f"{dataset_dir}/train", transform=train_transform)
    
    # Get targets (class labels) from the dataset
    targets = np.array(train_dataset.targets)
    
    # Perform stratified split
    train_indices, val_indices = train_test_split(
        np.arange(len(targets)),
        test_size=val_size,
        shuffle=True,
        stratify=targets,
        random_state=42
    )
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=f"{dataset_dir}/val", transform=val_test_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Get class names
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names
