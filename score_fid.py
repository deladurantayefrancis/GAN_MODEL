import argparse
import numpy as np
import os
import scipy
import torchvision
import torchvision.transforms as transforms
import torch
import classifier
from classifier import Classifier

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32

test_mu = None
test_sigma = None


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`
    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.
    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classifier.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size
    )
    return data_loader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def get_statistics(feature_iterator):

    # feature extraction from iterator
    imgs_feats = []
    try:
        while True:
            imgs_feats.append( next(feature_iterator) )
    except StopIteration:
        imgs_feats = np.asarray(imgs_feats)

    # retrieving statistics from images features
    mu = np.mean(imgs_feats, axis=0)
    sigma = np.cov(imgs_feats, rowvar=False)

    return mu, sigma


def get_fid_score(classifier, sampleloader, testloader):

    global test_mu, test_sigma
    
    if test_mu is None or test_sigma is None:
        test_feats = extract_features(classifier, testloader)
        test_mu, test_sigma = get_statistics(test_feats)
        #print('test_mu:', test_mu.shape)
        #print('test_sigma:', test_sigma.shape)

    sample_feats = extract_features(classifier, sampleloader)
    sample_mu, sample_sigma = get_statistics(sample_feats)

    mu_diff = test_mu - sample_mu
    cov_prod = scipy.linalg.sqrtm(test_sigma @ sample_sigma).real # keep real part only
    fid_score = mu_diff.dot(mu_diff) + np.trace(test_sigma + sample_sigma - 2*cov_prod)

    return fid_score
