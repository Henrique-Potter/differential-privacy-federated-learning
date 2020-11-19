import torch
import matplotlib.pyplot as plt

from util import load_mnist_dataset


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


# Loading the MNIST dataset
trainloader = load_mnist_dataset()

# make an iterator for looping
dataiter = iter(trainloader)

# Selecting the fist image from the dataset
images, labels = dataiter.next()

print(type(images))
print(images[0].shape)
# NOTE: The batch size is the number of images we get in one iteration

plt.imshow(images[1].numpy().squeeze())
plt.show()

# Creating a transformed view for the images (view - avoids copying the data)
flattened_images = images.view(64, 28 * 28)

# Flatten the images to shape(64, 784)
inputs = images.view(images.shape[0], -1)


# --- Manual creation of a NN single layer ---

# Defining the input layer with 256 neurons
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

# The output layer with the 10 classifications (0-10)
w2 = torch.randn(256, 10)
b2 = torch.randn(10)

# Multiplying the squeezed image input by the weights and adding the bias
inputAndWeights = torch.mm(inputs, w1) + b1
# Applying the activation function
h = sigmoid(torch.mm(inputs, w1) + b1)

# Multiplying layer 1 with the weights of the last layer to get a classification
out = torch.mm(h, w2) + b2

# Mapping the last 10 neurons output to equivalent probability ranges.
probabilities = softmax(out)

print(probabilities.shape)
print(probabilities)
print(probabilities.sum(dim=1))
