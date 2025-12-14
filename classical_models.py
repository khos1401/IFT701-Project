from torch import nn
import torch


class ClassicalCNN(nn.Module):
    """
    ClassicalCNN is a PyTorch module implementing a configurable feedforward neural network (FNN)
    for image classification.
    """

    def __init__(self,
                 input_size: list[int],
                 num_classes: int = 2,
                 hidden_dims = None,
                 dropout: float = 0.1):
        super(ClassicalCNN, self).__init__()

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout

        if hidden_dims is None:
            hidden_dims = [128]

        self.conv1 = nn.Conv2d(input_size[0], 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * input_size[1] // 4 * input_size[2] // 4, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_model_info(self):
        """Return model architecture and total parameters."""
        return {
            'model_type': 'ClassicalCNN',
            'input_size': self.input_size,
            'hidden_dims': self.hidden_dims,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture': str(self)
        }


class ClassicalNN(nn.Module):
    """
    ClassicalNN is a PyTorch module implementing a configurable feedforward neural network (FNN)
    for image classification (or general flattened inputs).
    """

    def __init__(self,
                 input_size: list[int],
                 num_classes: int = 2,
                 hidden_dims = None,
                 dropout: float = 0.1):
        super(ClassicalNN, self).__init__()

        if hidden_dims is None:
            hidden_dims = [516, 256]

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout

        self.flatten = nn.Flatten()

        # Build classification head from hidden_dims
        layers = []
        in_dim = torch.prod(torch.tensor(self.input_size)).item()

        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
            in_dim = h

        # Final output layer (logits for each class)
        layers.append(nn.Linear(in_dim, self.num_classes))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

        print(f'Classical Neural Network initialized:')
        print(f' - Input size (flattened): {input_size}')
        print(f' - Hidden dimensions: {hidden_dims}')
        print(f' - Number of classes: {num_classes}')
        print(f' - Dropout rate: {dropout}')
        print(f'Total parameters: {sum(p.numel() for p in self.parameters())}')
        print(f'Network architecture:')
        print(self)

    def _init_weights(self, m):
        """Initialize weights for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        """
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

    def get_model_info(self):
        """Return model architecture and total parameters."""
        return {
            'model_type': 'ClassicalNN',
            'input_size': self.input_size,
            'hidden_dims': self.hidden_dims,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture': str(self)
        }

