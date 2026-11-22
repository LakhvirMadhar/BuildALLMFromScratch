"""Test to see if pytorch properly installed.
https://pytorch.org/get-started/locally/
"""

if __name__ == "__main__":
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"My GPU: {torch.cuda.get_device_name(0)}\n")

    # Should print a tensor, test installation
    x = torch.rand(5, 3)
    print(x)
    print(type(x))

