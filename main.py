import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from model import FlexibleCNN
from dataset import get_data_loaders
from utils import print_model_analysis

def main(args):
    train_loader, val_loader, class_names = get_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print(f"Loaded dataset with {len(class_names)} classes: {class_names}")
    
    model = FlexibleCNN(
        input_channels=3,
        num_classes=len(class_names),
        num_filters=args.num_filters,
        filter_size=args.filter_size,
        activation_fn=nn.ReLU,
        dense_neurons=args.dense_neurons,
        input_size=args.img_size
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    # model analysis
    print_model_analysis(
        model=model,
        m=args.num_filters,
        k=args.filter_size,
        n=args.dense_neurons
    )
    
    # to verify model works
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # sample batch
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images)
        print(f"\nSample batch - Input shape: {images.shape}, Output shape: {outputs.shape}")
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible CNN for iNaturalist dataset")
    parser.add_argument("--train_dir", type=str, default="inaturalist_12k/train", help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="inaturalist_12k/val", help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_filters", type=int, default=10, help="Number of filters (m)")
    parser.add_argument("--filter_size", type=int, default=3, help="Filter size (k)")
    parser.add_argument("--dense_neurons", type=int, default=64, help="Number of neurons in dense layer (n)")
    
    args = parser.parse_args()
    main(args)
