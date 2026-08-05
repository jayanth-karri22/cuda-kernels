import torch
import numpy as np

data = [[1,2], [3,4]]

x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

shape=(2,3)

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

tensor = torch.rand(2,3)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(4,4)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")
tensor[:,1] = 0

t1 = torch.cat([tensor,tensor], dim=1)
t0 = torch.cat([tensor, tensor], dim=0)



y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
x = torch.matmul(tensor, tensor.T, out=y3)
print(x)

agg = tensor.sum()
agg_item = agg.item()

print(agg_item, type(agg_item))

print(tensor)

tensor.add_(5)
print(tensor)