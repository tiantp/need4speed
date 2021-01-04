# A hack to import models from parent directory
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from argparse import ArgumentParser
from torch.nn import functional as F
from models.SimpleCNN import SimpleCNN
from datamodules.cifar import Cifar10DataModule
from pytorch_lightning import LightningModule, Trainer, seed_everything, metrics

class Cifar10Experiment(LightningModule):

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                help='learning rate (default: 1e-3)')
        return parser

    def __init__(
        self,
        lr : float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.net = SimpleCNN()
        self.lr  = lr
        self.train_acc = metrics.Accuracy()
        self.val_acc = metrics.Accuracy()
        self.test_acc = metrics.Accuracy()

        self.save_hyperparameters()

    def _get_step_output(self, batch, batch_idx):
        x, y = batch
        log_prob = self.forward(x)
        loss = F.nll_loss(log_prob, y)
        y_hat = log_prob.argmax(dim=1, keepdim=True) # get index
        return {'x':x, 'y':y, 'y_hat':y_hat, 'loss':loss, 'log_prob':log_prob}

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        out = self._get_step_output(batch, batch_idx)
        self.log('train_loss', out['loss'])
        self.log('train_acc', self.train_acc(out['log_prob'], out['y']))
        return out['loss']

    def training_epoch_end(self, train_step_outpus):
        self.log('train_acc_epoch', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        out = self._get_step_output(batch, batch_idx)
        self.log('val_loss', out['loss'])
        self.log('val_acc', self.val_acc(out['log_prob'], out['y']))

    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_acc_epoch', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        out = self._get_step_output(batch, batch_idx)
        self.log('test_loss', out['loss'])
        self.log('test_acc', self.test_acc(out['log_prob'], out['y']))

    def test_epoch_end(self, test_step_outputs):
        self.log('test_acc_epoch', self.test_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def cli():
    seed_everything(42) # ensure reproducibility

    # Training settings
    DataModule = Cifar10DataModule
    Experiment = Cifar10Experiment

    parser = ArgumentParser(description='Cifar10 Example')
    parser = DataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()

    exp = Experiment(**vars(args))
    dm  = DataModule(**vars(args))
    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.fit(exp, dm)
    trainer.test()

    print("That's all folks")

if __name__ == "__main__":
    cli()

