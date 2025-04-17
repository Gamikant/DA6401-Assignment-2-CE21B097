from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_dir='inaturalist_12k/train', val_dir='inaturalist_12k/val', 
                     batch_size=32, img_size=224):
    """
    Create data loaders for the iNaturalist dataset
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for data loaders
        img_size: Size to resize images to
        
    Returns:
        train_loader, val_loader, class_names
    """
    # Defining transforms for training and validation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    
    class_names = train_dataset.classes
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, class_names
