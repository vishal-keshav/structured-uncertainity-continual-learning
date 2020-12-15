"""A dataset defines how data is created and consumed in train function.
All functionality must be contained in a dataset class. For example, torch
Dataset is a typical example of a dataset class, however, this can be more
flexible with custome dataset class with custome functions.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

# Example 1 (To create torch datasets from extrnal data sources)
"""
class dataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, idx):
        return 0
    def __len__(self):
        return 100
"""
# Example 2 (A custom dataset that includes both train and test)
class dataset:
    def __init__(self, data_path='../data'):
        mean, std = (0.5,), (1.0,)
        transformations = [transforms.ToTensor(),
                           transforms.Normalize(mean,std)]
        if not os.path.exists(data_path): os.makedirs(data_path)
        self.mnist_train = datasets.MNIST(data_path, train=True, download=True,
            transform=transforms.Compose(transformations))
        self.mnist_test = datasets.MNIST(data_path, train=False, download=True,
            transform=transforms.Compose(transformations))

    def get_data(self, batch_size):
        train_loader = torch.utils.data.DataLoader(self.mnist_train,
            batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(self.mnist_test,
            batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

################################# Test the dataset #############################
def test():
    from matplotlib import pyplot as plt
    d = dataset("../data")
    train_data, test_data = d.get_data(1)
    # Dataset is indexable but Dataloader is only iteratable, hence workers
    # use to fetch data in batches.
    for x, y in train_data:
        print(y[0])
        plt.imshow(x[0][0,:,:].numpy())
        plt.show()

if __name__ == "__main__":
    test()
