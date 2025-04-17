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
        
        # filter counts based on organization strategy
        if filter_org == 'same':
            filter_counts = [num_filters] * 5
        elif filter_org == 'double':
            filter_counts = [num_filters * (2**i) for i in range(5)]
        elif filter_org == 'half':
            filter_counts = [num_filters // (2**i) if num_filters // (2**i) >= 8 else 8 for i in range(5)]
        else:
            filter_counts = [num_filters] * 5
        
        # Building convolutional layers
        layers = []
        
        # First conv block
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
        
        # Calculating flattened size after convolutions
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
        
    def calculate_computations(self, m, k, n):
        """
        Calculate the total number of computations (multiply-adds) in the network.
        
        Args:
            m: Number of filters in each convolutional layer
            k: Size of filters (k×k)
            n: Number of neurons in the dense layer
            
        Returns:
            Total number of computations
        """
        input_size = self.input_size
        
        # First conv layer (3 input channels)
        comp_conv1 = input_size * input_size * m * (k * k * 3)
        
        # Second conv layer
        size2 = input_size // 2
        comp_conv2 = size2 * size2 * m * (k * k * m)
        
        # Third conv layer
        size3 = size2 // 2
        comp_conv3 = size3 * size3 * m * (k * k * m)
        
        # Fourth conv layer
        size4 = size3 // 2
        comp_conv4 = size4 * size4 * m * (k * k * m)
        
        # Fifth conv layer
        size5 = size4 // 2
        comp_conv5 = size5 * size5 * m * (k * k * m)
        
        # Final size after 5 max pooling layers
        final_size = size5 // 2
        
        # Dense layer
        comp_dense = n * (final_size * final_size * m)
        
        # Output layer
        comp_output = 10 * n
        
        total_comp = comp_conv1 + comp_conv2 + comp_conv3 + comp_conv4 + comp_conv5 + comp_dense + comp_output
        
        return total_comp
    
    def calculate_parameters(self, m, k, n):
        """
        Calculate the total number of parameters in the network.
        
        Args:
            m: Number of filters in each convolutional layer
            k: Size of filters (k×k)
            n: Number of neurons in the dense layer
            
        Returns:
            Total number of parameters
        """
        # First conv layer (3 input channels)
        params_conv1 = m * (k * k * 3 + 1)  # +1 for bias
        
        # Second conv layer
        params_conv2 = m * (k * k * m + 1)
        
        # Third conv layer
        params_conv3 = m * (k * k * m + 1)
        
        # Fourth conv layer
        params_conv4 = m * (k * k * m + 1)
        
        # Fifth conv layer
        params_conv5 = m * (k * k * m + 1)
        
        # Final size after 5 max pooling layers
        final_size = self.input_size // (2**5)
        
        # Dense layer
        params_dense = (final_size * final_size * m) * n + n
        
        # Output layer
        params_output = n * 10 + 10
        
        total_params = params_conv1 + params_conv2 + params_conv3 + params_conv4 + params_conv5 + params_dense + params_output
        
        return total_params

