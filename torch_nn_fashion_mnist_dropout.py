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


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


trainloader = load_fashion_mnist_testdataset()
testloader = load_fashion_mnist_testdataset()

# Initialize the network
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 20
steps = 0
train_losses, test_losses = [], []

for i in range(epochs):
    running_loss = 0
    for images, target_labels in trainloader:
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
            # set model to eval mode
            model.eval()
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # set model back to train mode
        model.train()

        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print(f'Training loss: {running_loss / len(trainloader)}')
        print(f'Test loss: {test_loss / len(testloader)}')
        print(f'Accuracy: {accuracy / len(testloader)}')

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)

# It's impractical to create new models every time we want to train or predict.
# To save we'll use the pytorch's torch.save().
# We can save to a file as follows:
torch.save(model.state_dict(), 'checkpoint.pth')

# load the saved model
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())

