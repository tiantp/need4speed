import PIL
import numpy as np
import time
import torch

from argparse import ArgumentParser
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, random_split
from typing import Optional, Callable, Tuple, Any
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10
from PIL import Image

# Use this class to apply normalization once as a preprocessing step, rather
# than for each epoch when looping through the data loader. But the benefits of
# apply preproc washes out quickly when we increase the number of workers for
# the dataset.
#
# Some benchmark (Running ResNet18 per epoch training time)
#            w Preproc   No Preproc (Regular CIFAR datasetModule)
#  1 workers : 10.649      12.664
#  2 workers : 10.721      10.605
# 16 workers : 10.927      10.756
class CIFAR10Normalized(CIFAR10) :
    def __init__(
              self,
              root: str,
              train: bool = True,
              transform: Optional[Callable] = None,
              target_transform: Optional[Callable] = None,
              download: bool = False,
      ) -> None:

        # Reading in CIFAR data from file
        start_time = time.time_ns()
        super(CIFAR10Normalized, self).__init__(root, train, transform, target_transform, download)
        elapsed = time.time_ns() - start_time
        print('Reading in data completed in {:0.3f} sec'.format(elapsed/1e9))

        # Normalize and whiten
        start_time = time.time_ns()
        mean=(0.4914, 0.4822, 0.4465)
        std=(0.2470, 0.2435, 0.2616)
        self.data = self.data / np.float32(255.0)
        self.data -= mean
        self.data /= std

        # Get into necessary shape to connect up with pytorch transforms
        self.data = self.data.transpose((0, 3, 1, 2)) # NHWC -> NCHW
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        self.data = torch.Tensor(self.data)

        elapsed = time.time_ns() - start_time
        print('Transforms (normalize, whiten) completed in {:0.3f} sec'.format(elapsed/1e9))

        print('self.data', type(self.data), type(self.data[0,0,0,0]),
                self.data.shape, self.data[0,:,0,0])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Cifar10DataModule(LightningDataModule):

    name = "cifar10"

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default="/tmp",
                metavar='DATA_DIR', help='data directory (default: "/tmp")')
        parser.add_argument('--train_batch_size', type=int, default=128,
                metavar='TRAIN_BATCH_SIZE', help='batch size for training (default: 128)')
        parser.add_argument('--val_batch_size', type=int, default=1000,
                metavar='VAL_BATCH_SIZE', help='batch size for validation (default: 1000)')
        parser.add_argument('--test_batch_size', type=int, default=1000,
                metavar='TEST_BATCH_SIZE', help='batch size for testing (default: 1000)')
        parser.add_argument('--augment_data', action='store_true', default=False,
                help='Apply data augmentation FlipLR and Random crop')
        parser.add_argument('--use_val', action='store_true', default=False,
                help='Partition the dataset into train and validation. This'
                'will reduce the amount of data for training. (default: False)')
        return parser


    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 128,
        test_batch_size: int = 256,
        val_batch_size: int = 256,
        augment_data: bool = False,
        use_val: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        self.augment_data = augment_data
        self.use_val = use_val



    def prepare_data(self):
        # Download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.augment_data :
                augment = Compose([RandomCrop(32, padding=4),
                            RandomHorizontalFlip()])
            # data_full = CIFAR10Normalized(self.data_dir, train=True,
            #         transform=augment if self.augment_data else [])
            # self.data_train, self.data_val = random_split(data_full,
            #     [train_sz, val_sz]) if self.use_val else (data_full, None)

            self.data_train = CIFAR10Normalized(self.data_dir, train=True,
                    transform=augment if self.augment_data else [])

            # Show CIFAR10 test accuracy after each epoch by hooking up through
            # the validation step
            self.data_val = CIFAR10Normalized(self.data_dir, train=False)

        if stage == 'test' or stage is None:
            self.data_test = CIFAR10Normalized(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size = self.train_batch_size,
                num_workers = 1, pin_memory = True, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size = self.test_batch_size,
            num_workers = 1, pin_memory = True, shuffle = False)

    def test_dataloader(self, transforms=None):
        return DataLoader(self.data_test, batch_size = self.val_batch_size,
                num_workers = 1, pin_memory = True, shuffle = False)


def test():
    print('Launching Test Scripts')
    dm = Cifar10DataModule()
    dm.prepare_data()

def time_train_dataloader():
    '''
    Prep Time ~ 1255 millisec
    Rough estimate of time for 1 worker, 1 epoch
    Normalize ~ 2633 millisec
    RandomCrop ~ 2893 millisec
    RandomFlipLR ~ 3425 millisec
    Using more workers can bring down the required time.
    '''
    import time
    start_time = time.time_ns()

    '''Amount of time required to run through one iteration of the dataset'''
    dm = Cifar10DataModule(augment_data=True)
    dm.prepare_data()
    dm.setup(stage='fit')
    dl = dm.train_dataloader()
    loop_start = time.time_ns()
    print('prep time {:0.4f} sec'.format( (loop_start - start_time)/1e9 ))
    epochs = 2
    print('Preparing to loop through {} epochs no_ops', epochs)
    for i in range(epochs) :
        for i, (batch, labels) in enumerate(dl): pass
    loop_end = time.time_ns()
    print('avg loop time {:0.4f} sec'.format( (loop_end - loop_start)/epochs/1e9 ))
    print('total time {:0.4f} sec'.format( (loop_end - start_time)/1e9 ))




if __name__ == "__main__" :
    # test()
    time_train_dataloader()
