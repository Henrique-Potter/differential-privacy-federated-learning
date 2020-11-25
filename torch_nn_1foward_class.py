from torch import nn
import torch.nn.functional as F


# Hardcoded neural net tuned for the MNIST dataset with two hidden layers
from util import load_mnist_traindataset


class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()
        # inputs to hidden layer linear transformation
        # 256 outputs
        self.hidden_layer1 = nn.Linear(784, 256)
        self.hidden_layer2 = nn.Linear(256, 64)
        # output layer, 10 units one for each digit
        self.output = nn.Linear(64, 10)

    def forward(self, x):

        # This hidden layer will receive the input, perform matrix multiplication and add a bias
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        x = self.hidden_layer1(x)

        # Relu activation
        first_layer_output = F.relu(x)

        x2 = self.hidden_layer2(first_layer_output)
        sec_layer_output = F.relu(x2)

        # Output layer with softmax activation
        y = F.softmax(self.output(sec_layer_output), dim=1)
        return y


# ---- Example application starts ----

# Loading the MNIST dataset
trainloader = load_mnist_traindataset()

# Get an instance of the MNISTNet class
model = MNISTNet()

# Prepare data
images, labels = next(iter(trainloader))

# Flatten images - This function will squeeze the 28x 28 matrices into 784
d1_images = images.view(images.shape[0], -1)

# forward pass, get the logits
logits = model(d1_images)
logist2 = model.forward(d1_images)

print(logits == logist2)

# Define a loss function
criterion = nn.CrossEntropyLoss()

# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)