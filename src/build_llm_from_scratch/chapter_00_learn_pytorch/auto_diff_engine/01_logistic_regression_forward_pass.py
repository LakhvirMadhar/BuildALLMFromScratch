import torch.nn.functional as F
import torch



if __name__ == "__main__":
    y = torch.tensor([1.0])     # True label
    x1 = torch.tensor([1.1])    # Input Feature
    w1 = torch.tensor([2.2])    # Weight Parameter
    b = torch.tensor([0.0])     # Bias Unit
    z = x1 * w1 + b             # Net input
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)

    print(f"z: {z}")
    print(f"a: {a}")
    print(f"loss: {loss}")
