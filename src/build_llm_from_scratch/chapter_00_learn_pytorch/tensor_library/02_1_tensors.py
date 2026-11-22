import torch

if __name__ == "__main__":
    print("Tensors of varying dimesnionality")
    tensor0d = torch.tensor(1)          # 0-d tensor (scalar)
    tensor1d = torch.tensor([1, 2, 3])  # 1-d tensor (vector)
    tensor2d = torch.tensor([           # 2-d tensor (matrix)
        [1,2],                          # 2D: one table, 4 rows -> shape (2, 2)
        [3,4]
    ])
    tensor3d = torch.tensor([           # 3-d tensor (3D tensor)
        [                               # 3D: two tables of 2 rows -> shape (2, 2, 2)
            [1,2], [3,4]
        ],
        [
            [5,6], [7,8]
        ]
    ]
    )
    tensors = [tensor0d, tensor1d, tensor2d, tensor3d]

    print(type(tensor0d))
    for t in tensors:
        print(t.shape)
