import torch

t = torch.tensor(1.0,requires_grad=True)
x = t * t
target = x * x
target.backward(retain_graph=True)
print(t.grad)
print(x.grad)
