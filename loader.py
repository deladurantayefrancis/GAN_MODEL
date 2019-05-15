import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split

def get_data_loader(dataset_name, batch_size):
    
    # retrieve dataset constructor
    if dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN
    elif dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100
    elif dataset_name == "stl10":
        dataset = torchvision.datasets.STL10
    elif dataset_name == "imagenet":
        dataset = torchvision.datasets.ImageNet
    
    # data normalization
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5),
                             (.5, .5, .5))
    ])
    
    # training and validation data 
    try:
        trainvalid = dataset(
            dataset_name, split='train+unlabeled',
            download=True,
            transform=image_transform
        )
    except:
        try:
            trainvalid = dataset(
                dataset_name, split='train',
                download=True,
                transform=image_transform
            )
        except:
            trainvalid = dataset(
                dataset_name, train=True,
                download=True,
                transform=image_transform
            )
    
    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )
    
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size
    )
    
    # test data
    try:
        testdata = dataset(
            dataset_name, split='test',
            download=True,
            transform=image_transform
        )
    except:
        testdata = dataset(
            dataset_name, train=False,
            download=True,
            transform=image_transform
        )
        
    testloader = torch.utils.data.DataLoader(
        testdata,
        batch_size=batch_size
    )
    
    return trainloader, validloader, testloader

