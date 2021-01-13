# A hack to import models from parent directory
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from argparse import ArgumentParser
from datamodules.cifar import Cifar10DataModule
from models.DawnNet import DawnNet
from models.SimpleCNN import SimpleCNN
from pytorch_lightning import LightningModule, Trainer, seed_everything, metrics
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR

class Cifar10Experiment(LightningModule):

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                help='learning rate (default: 1e-3) in `constant` scheduler.')
        parser.add_argument('--network', type=str, default='SimpleCNN',
                metavar='NETWORK',
                help='Network type, choose [SimpleCNN (default), DawnNet]')
        parser.add_argument('--lr_scheduler', type=str, default='constant',
                metavar='LR_SCHEDULER',
                help='[constant (default), onecycle] Constant learning rate or one cycle learning rate')
        parser.add_argument('--optimizer_name', type=str, default='adam',
                metavar='OPTIMIZER', help='[adam (default), sgd] optimizer')
        return parser

    def __init__(
        self,
        lr : float = 1e-3,
        network : str = 'SimpleCNN',
        lr_scheduler : str = 'constant', # `constant` or `onecycle`
        oc_max_lr : float = 0.1,# used for onecycle params
        oc_epochs : int = None,
        oc_steps_per_epoch : int = None,
        optimizer_name : str = 'adam',
        **kwargs,
    ):
        super().__init__()
        self.netname = network # faciliate tensorboard graph logging
        self.net = {'simplecnn':SimpleCNN(), 'dawnnet':DawnNet()}[network.lower()]
        self.lr  = lr
        self.lr_scheduler = lr_scheduler
        self.oc_max_lr = oc_max_lr
        self.oc_epochs = oc_epochs
        self.oc_steps_per_epoch = oc_steps_per_epoch
        self.optimizer_name = optimizer_name
        self.train_acc = metrics.Accuracy()
        self.val_acc   = metrics.Accuracy()
        self.test_acc  = metrics.Accuracy()

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
        self.log('train_accuracy', self.train_acc(out['log_prob'], out['y']))
        return out['loss']

    def training_epoch_end(self, train_step_outpus):
        self.log('train_accuracy_epoch_end', self.train_acc.compute())

        # Save graph to tensorboard
        if (self.current_epoch==0):
            self.logger.experiment.add_graph(
                    Cifar10Experiment(network=self.netname),
                    torch.rand((1,3,32,32)))

    def validation_step(self, batch, batch_idx):
        out = self._get_step_output(batch, batch_idx)
        self.log('val_loss', out['loss'])
        self.log('val_accuracy', self.val_acc(out['log_prob'], out['y']))

    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_accuracy_epoch_end', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        out = self._get_step_output(batch, batch_idx)
        self.log('test_loss', out['loss'])
        self.log('test_accuracy', self.test_acc(out['log_prob'], out['y']))

    def test_epoch_end(self, test_step_outputs):
        self.log('test_accuracy_epoch_end', self.test_acc.compute())

    def configure_optimizers(self):
        opt_cstor = {'adam':Adam, 'sgd':SGD }[self.optimizer_name.lower()]

        print('------->')
        print('------->')
        print('------->')
        print('------->')
        print('------->')

        if self.lr_scheduler.lower() == 'constant':
            optimizer = opt_cstor(self.parameters(), lr=self.lr)

            print('------> optimizer', optimizer)

            return optimizer

        elif self.lr_scheduler.lower() == 'onecycle':
            optimizers = [opt_cstor(self.parameters(), lr=self.lr)]
            epochs     = self.oc_epochs
            max_lr     = self.oc_max_lr
            steps_per_epoch = self.oc_steps_per_epoch
            initial_lr = 0.006
            div_factor = max_lr / initial_lr # inital_lr = max_lr / div_factor
            scheduler = OneCycleLR(optimizers[0], max_lr=max_lr,
                    epochs=self.oc_epochs, steps_per_epoch=steps_per_epoch,
                    verbose=True, anneal_strategy='linear', pct_start=0.5)
            schedulers = [{'scheduler':scheduler, 'interval':'epoch'}]

            print('------> optimizer', optimizers[0])

            return optimizers, schedulers

        else :
            raise ValueError('Error Cifar10Experiment::configure_optimizers(): '
                    'unrecognized scheduler `{}`'.format(self.lr_scheduler))

class PrintingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print('\n') # keep tqdm progress bar loss value from last epoch


def cli():
    seed_everything(42) # ensure reproducibility

    # Training settings
    DataModule = Cifar10DataModule
    Experiment = Cifar10Experiment

    parser = ArgumentParser(description='Cifar10 Example')
    parser = DataModule.add_argparse_args(parser)
    parser = Experiment.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()

    dm     = DataModule(**vars(args))
    dm.prepare_data()     # These two steps are needed to get num batches
    dm.setup(stage='fit') # in order to init the OneCycleLR params below
    exp    = Experiment(oc_max_lr=0.1, oc_epochs=args.max_epochs,
                oc_steps_per_epoch=len(dm.train_dataloader()), **vars(args))

    trainer = Trainer.from_argparse_args(args, deterministic=True,
            callbacks=[PrintingCallback()])

    trainer.fit(exp, dm)
    trainer.test()

    print("That's all folks")

if __name__ == "__main__":
    cli()

