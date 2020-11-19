from torch import nn
from util import load_mnist_dataset


# --- NN specialized  for the MNIST using nn sequential ---

# A sequential container. Modules will be added to it in the order they are passed in the constructor.
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
model = nn.Sequential(nn.Linear(784, 256),
                      nn.Sigmoid(),
                      nn.Linear(256, 64),
                      nn.Sigmoid(),
                      nn.Linear(64, 10)
                      )

# Loading the MNIST dataset
trainloader = load_mnist_dataset()

# Prepare data
images, labels = next(iter(trainloader))

# flatten images
d1_images = images.view(images.shape[0], -1)

# forward pass, get the logits
logits = model(d1_images)

# define the loss
criterion = nn.CrossEntropyLoss()
# calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)
