import torch


# Set requires_grad=True on a tensor when creating the tensor will set it for backpropagation
x = torch.randn(2,2, requires_grad=True)

# Take all elements to the power of 2
y = x** 2

# Calculate the mean of all elements from this Matrix
z = y.mean()
print(z)
# Calculate the loss from the mean for each element
z.backward()

print(x.grad)