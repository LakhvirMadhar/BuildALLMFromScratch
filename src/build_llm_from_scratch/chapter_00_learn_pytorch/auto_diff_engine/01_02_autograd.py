if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from torch.autograd import grad

    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 * b
    a = torch.sigmoid(z)

    loss = F.binary_cross_entropy(a,y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, w1, retain_graph=True)

    print(grad_L_w1)
    print(grad_L_b)

    # Takes care of calculation itself
    loss.backward() # type: ignore
    print(w1.grad)
    print(b.grad)
