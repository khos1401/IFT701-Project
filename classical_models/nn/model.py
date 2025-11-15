import torch
import torch.nn as nn

class ClassicalNN(nn.Module):
    """
    ClassicalNN is a PyTorch module implementing a configurable feedforward neural network (FNN)
    for image classification (or general flattened inputs).

    Args:
        input_size (int): Number of input features AFTER flattening.
                          For images: input_size = C * H * W (or H * W for grayscale).
        num_classes (int): Number of output classes.
        hidden_dims (list of int, optional): Sizes of hidden layers. Default: [256, 128].
        dropout (float, optional): Dropout rate applied after each hidden layer. Default: 0.3.

    Example (16x16 grayscale cats vs dogs):
        >>> model = ClassicalNN(input_size=16*16, num_classes=2)
        >>> x = torch.randn(32, 1, 16, 16)   # batch of 32 images (1,16,16)
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([32, 2])

    Example (32x32 RGB):
        >>> model = ClassicalNN(input_size=3*32*32, num_classes=2)
        >>> x = torch.randn(32, 3, 32, 32)
        >>> out = model(x)
    """

    def __init__(self,
                 input_size: int,
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
        in_dim = self.input_size

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

        x can be:
          - [batch, C, H, W]  (images)
          - [batch, H, W]
          - [batch, input_size] (already flattened)

        Returns:
            torch.Tensor: Class probabilities of shape [batch, num_classes].
        """
        x = self.flatten(x)          # -> [batch, input_size]
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
