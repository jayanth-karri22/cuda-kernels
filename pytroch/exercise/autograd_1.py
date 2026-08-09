import torch

x = torch.tensor([1., 2.], requires_grad=True)

q = x**3 + 2*x # dq/dx = 3*x**2 + 2
#q.sum().backward()
q.backward(gradient=torch.ones_like(q))

print(3*x**2 + 2 == x.grad)