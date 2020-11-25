# Common functions used throughout this project
import matplotlib.pyplot as plt
import numpy as np


def load_mnist_traindataset():
    import torch
    from torchvision import datasets, transforms

    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
    # download example data (if necessary) and load it to memory
    trainset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    return trainloader


def load_mnist_traindataset():
    import torch
    from torchvision import datasets, transforms

    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
    # download example data (if necessary) and load it to memory
    trainset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    return trainloader


def load_fashion_mnist_traindataset():
    import torch
    from torchvision import datasets, transforms

    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
    # download example data (if necessary) and load it to memory
    # Download and load training data
    trainset = datasets.FashionMNIST(
        'data/FASHION_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    return trainloader


def load_fashion_mnist_testdataset():
    import torch
    from torchvision import datasets, transforms

    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    # Download and load test data
    testset = datasets.FashionMNIST(
        'data/FASHION_MNIST_data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return  testloader


# Function for viewing an image and it's predicted classes.
def view_classification_mnist(img, ps):

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    plt.show()


def view_classification_famnist(img, probabilities):
    """Utility to imshow the image and its predicted classes."""
    ps = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels([
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle Boot'
    ], size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    plt.show()

