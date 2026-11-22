import torch

if __name__ == "__main__":
    tensor2d = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ]
    )
    print("Original shape, 2x3 matrix")
    print(tensor2d.shape)

    print("\nReshape the tensor, 3x2 matrix")
    print(tensor2d.reshape(3,2))

    print("\nThe more common command for tensor reshape to match numpy")
    print(tensor2d.view())

    print("\nTranspose tensor: that is, flip on its diagonal")
    print(tensor2d.T)

    print("\nMatrix Multiplication")
    print(tensor2d.matmul(tensor2d.T))

    print("\nAlternatively for matrix multiplication")
    print(tensor2d @ tensor2d.T)


    RASKA_NOTE = """The subtle difference between .view() and .reshape() in PyTorch lies in
    their handling of memory layout: .view() requires the original data to be contiguous
    and will fail if it isn't, whereas .reshape() will work regardless, copying the data if nec-
    essary to ensure the desired shape."""

    print(RASKA_NOTE)
