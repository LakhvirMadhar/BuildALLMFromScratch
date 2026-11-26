"""
As per Sebastian, the core of Pytorch is 3 steps:
1. Defining the dataset
2. Defining the model
3. Defining the training loop

In practice seems like:
1. Define the dataset
2. Define DataLoader
3. Define the model
4. Define the training loop
"""

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.features = X
        self.labels = y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self) -> int:
        return self.labels.shape[0]


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

type TensorPair = tuple[torch.Tensor, torch.Tensor]

def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader[TensorPair]) -> float:
    model.eval()
    correct = 0.0
    total_examples = 0

    for _, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare).item()
        total_examples += len(compare)

    return correct / total_examples



if __name__ == "__main__":
    # STEP 1: DEFINE THE DATA
    # Training data
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5],
    ]
    )

    y_train = torch.tensor([0, 0, 0, 1, 1])

    # Test data
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6]
    ])

    y_test = torch.tensor([0, 1])
    print("Shape and Type:")
    print(f"X_train : {X_train.shape, X_train.dtype}")
    print(f"y_train: {y_train.shape, y_train.dtype}")

    print(f"X_test: {X_test.shape, X_test.dtype}")
    print(f"y_test: {y_test.shape, y_test.dtype}")

    # STEP 2: DEFINE DATALOADER BETWEEN TRAIN AND TEST DATASETS
    # Create Custom Dataset class and instantiate
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    print(len(train_ds))  # Seems to be length og y_train
    print(len(test_ds))   # Seems to be length of y_test

    # Instantiate dataloaders
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,   # shuffle as y_train tesnore is three 0s and two 1s, in that order. Must shuffle to learn
        num_workers=0,  # small dataset, more workers not needed atm
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # Iterate over it, the training epoch
    for idx, (x,y) in enumerate(train_loader):
        print(f"Batch {idx+1}: {x, y}")

    # STEP 3: DEFINE MODEL
    torch.manual_seed(123)
    model = NeuralNetwork(num_features=2, num_classes=2)

    # STEP 4: RUN TRAINING LOOP
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.5
    )

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = f.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}")
            print(f" | Batch {batch_idx:03d}/{len(train_loader): 03d}")
            print(f" | Train loss: {loss: .2f}")

    print("\nModel Eval:")
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
    print(outputs)

    print("\nClass Membership probabilities, what are probabilites of belong to class 0 and class 1")
    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    print(probas)

    predictions = torch.argmax(probas, dim=1)
    print(f"\n{predictions}")

    predictions = torch.argmax(outputs, dim=1)
    print(f"\n{predictions}")

    # Compare prediction to y_train
    print(predictions == y_train)
    print(torch.sum(predictions==y_train))

    # Compute accuracy
    print(compute_accuracy(model, train_loader))
    print(compute_accuracy(model, test_loader))

    # Save a model
    print("Saving model:")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved:")

    print("Loading Model:")
    model = NeuralNetwork(2,2)
    model.load_state_dict(torch.load("model.pth"))
    print("Model loaded!")