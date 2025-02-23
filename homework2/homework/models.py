"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        # Using CrossEntropyLoss for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return self.loss_fn(logits, target)
        #raise NotImplementedError("ClassificationLoss.forward() is not implemented")


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        # Flatten layer to reshape image (b, 3, H, W) -> (b, 3*h*w)
        self.flatten = nn.Flatten()

        # Linear layer to map input to num_classes
        self.fc = nn.Linear(3 * h * w, num_classes)
        #raise NotImplementedError("LinearClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten input tensor
        x = self.flatten(x)

        # Forward pass through linear layer
        logits = self.fc(x)

        return logits
        #raise NotImplementedError("LinearClassifier.forward() is not implemented")
    if __name__ == "__main__":
        # Example usage:
        batch_size = 4
        h, w = 64, 64
        num_classes = 6
    
    # Create dummy input tensor simulating a batch of images
        dummy_input = torch.randn(batch_size, 3, h, w)
    
    # Initialize the model
        model = LinearClassifier(h, w, num_classes)
    
    # Perform a forward pass
        logits = model(dummy_input)
    
        print("Input shape:", dummy_input.shape)   # Expected: (4, 3, 64, 64)
        print("Output logits shape:", logits.shape)  # Expected: (4, 6)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
            hidden_dim: int, number of neurons in the hidden layer
        """
        super().__init__()

        # Flatten layer to reshape input tensor
        self.flatten = nn.Flatten()

        # Fully connected layer from input to hidden layer
        self.fc1 = nn.Linear(3 * h * w, hidden_dim)

        # Activation function (ReLU)
        self.relu = nn.ReLU()

        # Fully connected layer from hidden layer to output layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten input
        x = self.flatten(x)

        # Forward pass: Input -> Hidden layer -> ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # Forward pass: Hidden layer -> Output layer
        logits = self.fc2(x)

        return logits
        #raise NotImplementedError("LinearClassifier.forward() is not implemented")
        
class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int, number of output classes
            hidden_dim: int, number of neurons in each hidden layer
            num_layers: int, number of hidden layers
        """
        super().__init__()

        # Flatten layer to reshape input tensor
        self.flatten = nn.Flatten()

        # Input layer
        layers = [nn.Linear(3 * h * w, hidden_dim), nn.ReLU()]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten input
        x = self.flatten(x)

        # Forward pass through all layers
        logits = self.model(x)

        return logits
        #raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        """
        An MLP with multiple hidden layers and residual connections.

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int, number of output classes
            hidden_dim: int, number of neurons in each hidden layer
            num_layers: int, number of hidden layers
        """
        super().__init__()

        # Flatten layer to reshape input tensor
        self.flatten = nn.Flatten()

        # Input layer
        self.input_layer = nn.Linear(3 * h * w, hidden_dim)

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.relu = nn.ReLU()

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten input
        x = self.flatten(x)

        # Input layer
        x = self.input_layer(x)
        x = self.relu(x)

        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            residual = x  # Store input for residual connection
            x = layer(x)
            x = self.relu(x)
            x += residual  # Add residual connection

        # Output layer
        logits = self.output_layer(x)

        return logits
        #raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module - The PyTorch model whose size is to be calculated.

    Returns:
        float - Size of the model in megabytes (MB).
    """
    # Count the total number of parameters (elements) in the model
    total_params = sum(p.numel() for p in model.parameters())

    # Each parameter typically takes 4 bytes (32-bit float)
    # Convert total bytes to megabytes by dividing by 1024 twice
    size_in_mb = total_params * 4 / 1024 / 1024

    return size_in_mb

def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
