import torch

if __name__ == "__main__":
    tensor1d = torch.tensor([1,2,3])
    print(tensor1d.dtype)

    float_vec = torch.tensor([1.0, 2.0, 3.0])
    print(float_vec.dtype)

    # Convert 64 bit to 32 bit
    float_vec_02 = tensor1d.to(torch.float32)
    print(float_vec_02.dtype)
