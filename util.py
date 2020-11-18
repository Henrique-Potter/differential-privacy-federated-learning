# Common functions used throughout this project


def load_mnist_dataset():
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
