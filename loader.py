import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split

def get_data_loader(dataset_name, category, batch_size):
    
    # retrieve dataset constructor and category id
    if dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN
        category_id = int(category)
    elif dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10
        category_id = ['plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'].index(category)
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100
        category_id = int(category)
    elif dataset_name == "imagenet":
        dataset = torchvision.datasets.ImageNet
        category_id = int(category)
    
    # data normalization
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5),
                             (.5, .5, .5))
    ])
    
    # training and validation data 
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
    
    print(dir(trainvalid))
    category_indices = np.where(np.array(trainvalid.labels) == category_id)[0]
    category_data = trainvalid.data[category_indices]
    category_labels = np.array(trainvalid.labels)[category_indices].tolist()
    trainvalid.data = category_data
    trainvalid.labels = category_labels
    
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
        batch_size=batch_size,
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
        ),
    
    print(dir(testdata))
    category_indices = np.where(np.array(testdata.labels) == category_id)[0]
    category_data = testdata.data[category_indices]
    category_labels = np.array(testdata.labels)[category_indices].tolist()
    testdata.data = category_data
    testdata.labels = category_labels
        
    testloader = torch.utils.data.DataLoader(
        testdata,
        batch_size=batch_size,
    )
    
    return trainloader, validloader, testloader

