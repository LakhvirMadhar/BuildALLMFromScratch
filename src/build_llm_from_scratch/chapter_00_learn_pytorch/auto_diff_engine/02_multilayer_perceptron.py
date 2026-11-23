import torch


class NeuralNetwork(torch.nn.Module):
    """A multilayer perceptron classifier with two hidden layers (Listing A.4).

    Each hidden layer is a fully connected `Linear` layer followed by a ReLU.
    The output layer returns raw scores (logits), one per class, with no
    activation applied.
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        """Build the layers.

        Args:
            num_features: How many input values each example has.
            num_classes: How many classes to score, one output per class.
        """
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of examples through the network.

        Args:
            x: Input of shape (batch_size, num_features).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        logits: torch.Tensor = self.layers(x)
        return logits


if __name__ == "__main__":
    model = NeuralNetwork(num_features=50, num_classes=3)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable model parameters: {num_params}")

    print("\nFirst layer contents at position 0")
    print(model.layers[0].weight)
    print(model.layers[0].weight.shape)

    torch.manual_seed(123)
    X = torch.rand((1,50))
    out = model(X)
    print(out)
