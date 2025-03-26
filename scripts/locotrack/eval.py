import sys
sys.path.append('../../repos/locotrack/locotrack_pytorch')

import os
import configparser
import argparse
import logging
from functools import partial
from typing import Any, Dict, Optional, Union, Tuple

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
import torch
from torch.utils.data import DataLoader

from models.locotrack_model import LocoTrack
import model_utils
from mydataset import get_eval_dataset


class LocoTrackModel(L.LightningModule):
    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_forward_kwargs: Optional[Dict[str, Any]] = None,
        loss_name: Optional[str] = 'tapir_loss',
        loss_kwargs: Optional[Dict[str, Any]] = None,
        query_first: Optional[bool] = False,
        optimizer_name: Optional[str] = 'Adam',
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_name: Optional[str] = 'OneCycleLR',
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = LocoTrack(**(model_kwargs or {}))
        self.model_forward_kwargs = model_forward_kwargs or {}
        self.loss = partial(model_utils.__dict__[loss_name], **(loss_kwargs or {}))
        self.query_first = query_first

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 2e-3}
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs or {'max_lr': 2e-3, 'pct_start': 0.05, 'total_steps': 300000}

    def training_step(self, batch, batch_idx):
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(batch, output)
        
        self.log_dict(
            {f'train/{k}': v.item() for k, v in loss_scalars.items()},
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(batch, output)
        metrics = model_utils.eval_batch(batch, output, query_first=self.query_first)
        
        log_prefix = 'val/'
        if dataloader_idx is not None:
            log_prefix = f'val/data_{dataloader_idx}/'

        self.log_dict(
            {log_prefix + k: v for k, v in loss_scalars.items()},
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(batch, output)
        metrics = model_utils.eval_batch(batch, output, query_first=self.query_first)

        log_prefix = 'test/'
        if dataloader_idx is not None:
            log_prefix = f'test/data_{dataloader_idx}/'
        
        self.log_dict(
            {log_prefix + k: v for k, v in loss_scalars.items()},
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")
        
    def configure_optimizers(self):
        weights = [p for n, p in self.named_parameters() if 'bias' not in n]
        bias = [p for n, p in self.named_parameters() if 'bias' in n]

        optimizer = torch.optim.__dict__[self.optimizer_name](
            [
                {'params': weights, **self.optimizer_kwargs},
                {'params': bias, **self.optimizer_kwargs, 'weight_decay': 0.}
            ]
        )
        scheduler = torch.optim.lr_scheduler.__dict__[self.scheduler_name](optimizer, **self.scheduler_kwargs)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train(
    mode: str,
    save_path: str,
    val_dataset_path: str,
    ckpt_path: str = None,
    kubric_dir: str = '',
    precision: str = '32',
    batch_size: int = 1,
    val_check_interval: Union[int, float] = 5000,
    log_every_n_steps: int = 10,
    gradient_clip_val: float = 1.0,
    max_steps: int = 300_000,
    model_kwargs: Optional[Dict[str, Any]] = None,
    model_forward_kwargs: Optional[Dict[str, Any]] = None,
    loss_name: str = 'tapir_loss',
    loss_kwargs: Optional[Dict[str, Any]] = None,
    optimizer_name: str = 'Adam',
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    scheduler_name: str = 'OneCycleLR',
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    # query_first: bool = False,
    proportions: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0),
    image_size: Tuple[int, int] = (256, 256),
):
    """Train the LocoTrack model with specified configurations."""
    seed_everything(42, workers=True)

    model = LocoTrackModel(
        model_kwargs=model_kwargs,
        model_forward_kwargs=model_forward_kwargs,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        query_first='first' in mode,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
    )
    if ckpt_path is not None and 'train' in mode:
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    # logger = WandbLogger(project='LocoTrack_Pytorch', save_dir=save_path, id=os.path.basename(save_path))
    logger = CSVLogger(save_dir=save_path)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_last=True,
        save_top_k=3,
        mode="max",
        monitor="val/average_pts_within_thresh",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
    )

    eval_dataset = get_eval_dataset(
        mode=mode,
        path=val_dataset_path,
        resolution=image_size,
        proportions=proportions,
    )
    eval_dataloder = {
        k: DataLoader(
            v,
            batch_size=1,
            shuffle=False,
        ) for k, v in eval_dataset.items()
    }
    
    trainer = L.Trainer(strategy='ddp', logger=logger, precision=precision)
    trainer.test(model, eval_dataloder, ckpt_path=ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate the LocoTrack model.")
    parser.add_argument('--config', type=str, default='config.ini', help="Path to the configuration file.")
    parser.add_argument('--mode', type=str, required=True, help="Mode for training or evaluation.")
    parser.add_argument('--data_root', type=str, default='data', help="Root directory for the data.")
    parser.add_argument('--proportions', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="Proportions for train, val, and test datasets.")
    parser.add_argument('--image_size', type=int, default=[256, 256], nargs=2, help="Size of the input images.")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument('--save_path', type=str, default='snapshots', help="Path to save the logs and checkpoints.")
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Extract parameters from the config file
    train_params = {
        'mode': args.mode,
        'ckpt_path': args.ckpt_path,
        'save_path': args.save_path,
        # 'val_dataset_path': eval(config.get('TRAINING', 'val_dataset_path', fallback='{}')),
        'val_dataset_path': args.data_root,
        'proportions': tuple(args.proportions),
        'image_size': tuple(args.image_size),
        'kubric_dir': config.get('TRAINING', 'kubric_dir', fallback=''),
        'precision': config.get('TRAINING', 'precision', fallback='32'),
        'batch_size': config.getint('TRAINING', 'batch_size', fallback=1),
        'val_check_interval': config.getfloat('TRAINING', 'val_check_interval', fallback=5000),
        'log_every_n_steps': config.getint('TRAINING', 'log_every_n_steps', fallback=10),
        'gradient_clip_val': config.getfloat('TRAINING', 'gradient_clip_val', fallback=1.0),
        'max_steps': config.getint('TRAINING', 'max_steps', fallback=300000),
        'model_kwargs': eval(config.get('MODEL', 'model_kwargs', fallback='{}')),
        'model_forward_kwargs': eval(config.get('MODEL', 'model_forward_kwargs', fallback='{}')),
        'loss_name': config.get('LOSS', 'loss_name', fallback='tapir_loss'),
        'loss_kwargs': eval(config.get('LOSS', 'loss_kwargs', fallback='{}')),
        'optimizer_name': config.get('OPTIMIZER', 'optimizer_name', fallback='Adam'),
        'optimizer_kwargs': eval(config.get('OPTIMIZER', 'optimizer_kwargs', fallback='{"lr": 2e-3}')),
        'scheduler_name': config.get('SCHEDULER', 'scheduler_name', fallback='OneCycleLR'),
        'scheduler_kwargs': eval(config.get('SCHEDULER', 'scheduler_kwargs', fallback='{"max_lr": 2e-3, "pct_start": 0.05, "total_steps": 300000}')),
    }

    train(**train_params)
