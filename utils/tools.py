
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

def get_cifar10_mean_std():
    '''
    mean [0.4914, 0.4822, 0.4465]
    std  [0.2470, 0.2435, 0.2616]
    '''
    count     = 0
    x_sum     = 0.0
    x_squared = 0.0
    data      = CIFAR10('/tmp/', train=True, download=True,
                    transform=transforms.ToTensor())
    loader    = DataLoader(data, batch_size = 128,
                    num_workers = 12, pin_memory = True, shuffle = True)
    for index, (x, _) in enumerate(loader):
        x_sum     += x.sum([0, 2, 3]) # reduce all dim except channels
        x_squared += (x ** 2).sum([0, 2, 3])
        count     += np.prod( x.shape ) / 3  # batch * pixels

    mean = x_sum / count
    var  = x_squared / count - mean ** 2
    std  = np.sqrt(var)

    print('x_sum', x_sum, 'x_squared', x_squared, 'count', count)
    print('mean', mean, 'var', var, 'std', std)

    return {'mean':mean, 'std':std}


if __name__ == '__main__':
    print(get_cifar10_mean_std())
