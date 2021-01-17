# Adapted from example in
#    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

import torch
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST

class MnistDataModule(LightningDataModule):

    name = "mnist"

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default="./",
                metavar='N', help='data directory (default: "./")')
        parser.add_argument('--train_batch_size', type=int, default=64,
                metavar='N', help='batch size for training (default: 64)')
        parser.add_argument('--val_batch_size', type=int, default=1000,
                metavar='N', help='batch size for validation (default: 1000)')
        parser.add_argument('--test_batch_size', type=int, default=1000,
                metavar='N', help='batch size for testing (default: 1000)')
        return parser


    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 64,
        test_batch_size: int = 1000,
        val_batch_size: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    def prepare_data(self):
        # Download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            data_full = MNIST(self.data_dir, train=True,
                    transform=self.transform)
            self.data_train, self.data_val = random_split(
                    data_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.data_test = MNIST(self.data_dir, train=False,
                    transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size = self.train_batch_size,
                num_workers = 12, pin_memory = True, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size = self.test_batch_size,
                num_workers = 12, pin_memory = True, shuffle = False)

    def test_dataloader(self, transforms=None):
        return DataLoader(self.data_test, batch_size = self.val_batch_size,
                num_workers = 12, pin_memory = True, shuffle = False)

