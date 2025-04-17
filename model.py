import torch
import torch.nn as nn

class FlexibleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, num_filters=32, filter_size=3,
                 activation_fn=nn.ReLU, dense_neurons=128, input_size=224, 
                 filter_org='same', use_batchnorm=False, dropout_rate=0):
        super(FlexibleCNN, self).__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dense_neurons = dense_neurons
        self.input_size = input_size
        self.filter_org = filter_org
        
        # Determine filter counts based on organization strategy
        if filter_org == 'same':
            filter_counts = [num_filters] * 5
        elif filter_org == 'double':
            filter_counts = [num_filters * (2**i) for i in range(5)]
        elif filter_org == 'half':
            filter_counts = [num_filters // (2**i) if num_filters // (2**i) >= 8 else 8 for i in range(5)]
        else:
            # Default to same
            filter_counts = [num_filters] * 5
        
        # Build convolutional layers
        layers = []
        
        # First conv block (input channels -> filter_counts[0])
        layers.append(nn.Conv2d(input_channels, filter_counts[0], kernel_size=filter_size, padding=filter_size//2))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(filter_counts[0]))
        layers.append(activation_fn())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Remaining conv blocks
        for i in range(1, 5):
            layers.append(nn.Conv2d(filter_counts[i-1], filter_counts[i], kernel_size=filter_size, padding=filter_size//2))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(filter_counts[i]))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        self.output_size = input_size // (2**5)
        self.flatten_size = self.output_size * self.output_size * filter_counts[4]
        
        # Dense layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, dense_neurons),
            activation_fn(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc_layers(x)
        return x
