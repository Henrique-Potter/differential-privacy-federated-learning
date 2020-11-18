import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def sigmoid(x):
    """Create a sigmoid activation function.
    Good for outputs that fall between 0 and 1. (probability)
    args x: a torch tensor.
    """
    return 1/(1 + torch.exp(-x))


# This function will map the values and distribute then on a probability range according to the highest and lowest
# value.
def softmax(x):
    """Create a softmax activation function.
    Good for outputs that fall between 0 and 1. (probability)
    args x: a torch tensor.
    """
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)


# define a transform to normalize the data
# if the img has three channels, you should have three number for mean,
# for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
# If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                               ])

# download and load the traning data
trainset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# make an iterator for looping
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images[0].shape)
# NOTE: The batch size is the number of images we get in one iteration

plt.imshow(images[1].numpy().squeeze())
plt.show()

# Creating a transformed view for the images (avoids copying the data)
flattened_images = images.view(64, 28 * 28)

# flatten the images to shape(64, 784)
inputs = images.view(images.shape[0], -1)

# create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = sigmoid(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2
probabilities = softmax(out)

print(probabilities.shape)
print(probabilities)
print(probabilities.sum(dim=1))
