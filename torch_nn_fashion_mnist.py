import torch

from torch import nn
from util import load_mnist_traindataset, view_classification_mnist, load_fashion_mnist_testdataset, \
    view_classification_famnist
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

input_units = 784
hidden_units = [256, 128, 64]
output_units = 10

# 3 Layer neural network
model = nn.Sequential(
    nn.Linear(input_units, hidden_units[0]),
    nn.ReLU(),
    nn.Linear(hidden_units[0], hidden_units[1]),
    nn.ReLU(),
    nn.Linear(hidden_units[1], hidden_units[2]),
    nn.ReLU(),
    nn.Linear(hidden_units[2], output_units),
    nn.LogSoftmax(dim=1)
)

# Using Adam optimization. Computationally more efficient and performs better with non-stationary objects.
# Has dynamic learning rate.
# Good for noisy data.
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

trainloader = load_fashion_mnist_testdataset()
testloader = load_fashion_mnist_testdataset()

epochs = 3

# Train the network
for i in range(epochs):
    running_loss = 0
    for images, target_labels in trainloader:
        # flatten images into 784 long vector for the input layer
        images = images.view(images.shape[0], -1)
        # clear gradients because they accumulate
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, target_labels)
        # let optmizer update the parameters
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f'Training loss: {running_loss / len(trainloader)}')


dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]
# reshape images to the model input layer's liking.
# get class probabilities (10 class probabilities for 64 examples)
test_images = images.view(images.shape[0], -1)
# Calculate class probabilities (softmax)
ps = torch.exp(model(test_images))
print(ps.shape)

# plot out the image and probability distribution
view_classification_famnist(img, ps[0])


images, labels = next(iter(testloader))
# get class probabilities (10 class probabilities for 64 examples)
images = images.view(images.shape[0], -1)
ps = torch.exp(model(images))
ps.shape

# With the probabilities, we can use ps.topk to get the most likely class and return the k highest values.
# Since we just want the most likely class, we can use ps.topk(1).
# This returns a tuple of top-k values and top-k indices.
# If the highest values is the 5th element, we'll get back 4 as the index.
top_p, top_class = ps.topk(1, dim=1)
print(top_class.shape)
print(labels.shape)

# Check where our predicted classes match with true classes from labels
equals = top_class == labels.view(*top_class.shape) # make sure they have the same shape
# Convert the equals byte tensor into float tensor before doing the mean
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')


# Train the network
epochs = 30
steps = 0

train_losses, test_losses = [], []
for i in range(epochs):
    running_loss = 0
    for images, target_labels in trainloader:
        # flatten images into 784 long vector for the input layer
        images = images.view(images.shape[0], -1)
        # clear gradients because they accumulate
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, target_labels)
        # let optmizer update the parameters
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        accuracy = 0
        test_loss = 0

        # turn off gradients for validation, saves memory and computation
        with torch.no_grad():
            for images, labels in testloader:
                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print(f'Accuracy: {accuracy / len(testloader)}')
        print(f'Training loss: {running_loss / len(trainloader)}')
        print(f'Test loss: {test_loss / len(testloader)}')


import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()