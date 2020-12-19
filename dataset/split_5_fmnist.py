import numpy as np
from tqdm import tqdm
import torch
from torchvision import datasets
from torchvision import transforms

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

class dataset:
    def __init__(self, data_path):
        mean = (0.1307,)
        std = (0.3081,)
        transformations = [transforms.ToTensor(),
                           transforms.Normalize(mean,std)]
        mnist_train = datasets.FashionMNIST(data_path, train=True, download=True,
            transform=transforms.Compose(transformations))
        mnist_test = datasets.FashionMNIST(data_path, train=False, download=True,
            transform=transforms.Compose(transformations))
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1,
            shuffle=False)
        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1,
            shuffle=False)

        self.train_task_data = {i:{'x': [], 'y': []} for i in range(5)}
        for x, y in tqdm(train_loader):
            y_npy = y.numpy()[0]
            task_id, label = y_npy//2, y_npy%2
            self.train_task_data[task_id]['x'].append(x)
            self.train_task_data[task_id]['y'].append(label)

        self.test_task_data = {i:{'x': [], 'y': []} for i in range(5)}
        for x, y in tqdm(test_loader):
            y_npy = y.numpy()[0]
            task_id, label = y_npy//2, y_npy%2
            self.test_task_data[task_id]['x'].append(x)
            self.test_task_data[task_id]['y'].append(label)

        for task_id in tqdm(range(5)):
            self.train_task_data[task_id]['x'] = torch.stack(
                self.train_task_data[task_id]['x']).view(-1, 1, 28, 28)
            self.train_task_data[task_id]['y'] = torch.LongTensor(
                np.array(self.train_task_data[task_id]['y'], dtype=int)).view(-1)
            self.test_task_data[task_id]['x'] = torch.stack(
                self.test_task_data[task_id]['x']).view(-1, 1, 28, 28)
            self.test_task_data[task_id]['y'] = torch.LongTensor(
                np.array(self.test_task_data[task_id]['y'], dtype=int)).view(-1)


    def get_train_split(self, task_id):
        return (self.train_task_data[task_id]['x'],
            self.train_task_data[task_id]['y'])

    def get_validation_split(self, task_id):
        return self.get_train_split(task_id)

    def get_test_split(self, task_id):
        return (self.test_task_data[task_id]['x'],
            self.test_task_data[task_id]['y'])

######################################## TEST ##################################
def test():
    from matplotlib import pyplot as plt
    d = dataset("../data")
    for task_id in range(5):
        x, y = d.get_train_split(task_id)
        for i in range(3):
            print(y[i])
            plt.imshow(x[i][0,:,:])
            plt.show()
        x, y = d.get_test_split(task_id)
        for i in range(3):
            print(y[i])
            plt.imshow(x[i][0,:,:])
            plt.show()

if __name__ == "__main__":
    test()