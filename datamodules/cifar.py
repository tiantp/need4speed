import torch
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10

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
        tlist= [transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))]

        # Assign train/val dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.augment_data :
                tlist.append(transforms.RandomCrop(32, padding=4))
                tlist.append(transforms.RandomHorizontalFlip())
            data_full = CIFAR10(self.data_dir, train=True,
                    transform=transforms.Compose(tlist))
            self.data_train, self.data_val = random_split(data_full,
                [train_sz, val_sz]) if self.use_val else (data_full, None)

        if stage == 'test' or stage is None:
            self.data_test = CIFAR10(self.data_dir, train=False,
                    transform=transforms.Compose(tlist))


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size = self.train_batch_size,
                num_workers = 4, pin_memory = True, shuffle = True)

    def val_dataloader(self):
        if self.data_val :
            return DataLoader(self.data_val, batch_size = self.test_batch_size,
                num_workers = 4, pin_memory = True, shuffle = False)
        else :
            return None

    def test_dataloader(self, transforms=None):
        return DataLoader(self.data_test, batch_size = self.val_batch_size,
                num_workers = 4, pin_memory = True, shuffle = False)


def test():
    print('Launching Test Scripts')
    dm = Cifar10DataModule()
    dm.prepare_data()


if __name__ == "__main__" :
    test()
