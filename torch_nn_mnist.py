import torch

from torch import nn
from util import load_mnist_traindataset, view_classification_mnist
from torch import optim
import torch.nn.functional as F
import numpy as np

# Making the results deterministic
torch.manual_seed(254)
np.random.seed(254)

# --- NN specialized  for the MNIST using nn sequential ---

# A sequential container. Modules will be added to it in the order they are passed to the constructor.
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
# Here we are using ReLU activations since is faster to calculate and often offers better results
# LogSoftmax will give us probabilities at the output layer
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1),
                     )

# Negative log likelihood as our loss.
criterion = nn.NLLLoss()

# Loading the MNIST dataset
trainloader = load_mnist_traindataset()

# Prepare data
images, labels = next(iter(trainloader))

# We also need an optimizer that'll update weights with the gradients from the backward pass.
# From Pytorch's optim package, we can use stochastic gradient descenc with optim.SGD
# Pass in the parameter to optimize and a learning rate "lr"
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 6
for e in range(epochs):
    running_loss = 0

    for images, labels in trainloader:
        # Flatten Images into 784 long vector for the input layer
        images = images.view(images.shape[0], -1)

        # Clear gradients otherwise they accumulate with each iteration
        optimizer.zero_grad()
        # forward pass
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Printing the current loss for this iteration
    print(f'Training loss: {running_loss / len(trainloader)}')

img = images[0].view(1, 784)

# View classification
with torch.no_grad():
    logits = model.forward(img)

    ps = F.softmax(logits, dim=1)
    print(ps)
    view_classification_mnist(img.view(1, 28, 28), ps)


