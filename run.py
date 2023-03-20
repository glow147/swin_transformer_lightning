from utils.utility import load_setting
from torch.utils.data import DataLoader
from models.swin_transformer import SwinTransformer
from datasets.datasets import imagenet_train_dataset, imagenet_valid_dataset, custom_collate
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.loggers import WandbLogger

import torch
import random
import argparse
import numpy as np
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def train(cfg, model, train_dataloader, valid_dataloader):
    profiler = SimpleProfiler(dirpath='.', filename='model_logs')
    wandb_logger = WandbLogger(config=cfg)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='acc',
        dirpath=f'{cfg.MODEL.NAME}',
        filename='{epoch:02d}-{acc:.3f}',
        save_top_k=3,
        mode="max"
    )
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(accelerator='gpu', devices=torch.cuda.device_count(), max_epochs=300,
                         num_sanity_val_steps=0, callbacks=[ckpt_callback, lr_callback],
                         strategy=strategy if torch.cuda.device_count() > 1 else None,
                         precision=16, profiler=profiler, benchmark=True, gradient_clip_val=5, gradient_clip_algorithm='norm',
                         logger=wandb_logger)

    rank = 0 + trainer.local_rank
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="configs/swin_tiny_patch4_window7_224.yaml")

    args = parser.parse_args()

    cfg = load_setting(args.setting)

    train_collate_fn = custom_collate(cfg, is_train=True)
    train_dataset = imagenet_train_dataset(cfg)
    train_dataloader = DataLoader(train_dataset, prefetch_factor=4, shuffle=True, batch_size=512, num_workers=16, pin_memory=True, collate_fn=train_collate_fn)

    valid_collate_fn = custom_collate(cfg, is_train=False)
    valid_dataset = imagenet_valid_dataset(cfg, train_dataset.get_token())
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=512, num_workers=8, pin_memory=True, collate_fn=valid_collate_fn)

    model = SwinTransformer(cfg = cfg,
                            img_size=cfg.DATA.IMAGE_SIZE,
                            patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
                            in_chans=cfg.MODEL.SWIN.IN_CHANS,
                            num_classes=cfg.MODEL.NUM_CLASSES,
                            embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
                            num_heads=cfg.MODEL.SWIN.NUM_HEADS,
                            depths=cfg.MODEL.SWIN.DEPTHS,
                            window_size=cfg.MODEL.SWIN.WINDOW_SIZE)

    model = model.to(device)

    train(cfg, model, train_dataloader, valid_dataloader)
