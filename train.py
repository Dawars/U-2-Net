import os

import torch
import lightning.pytorch as pl
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.optim as optim

from data_loader import DocProjDataset, RescaleT, RandomCrop, ToTensorLab, DIWDataset, Doc3DDataset

from model import U2NET, U2NETP
from u2net_test import normPRED

docproj_root = "/mnt/hdd/datasets/documents/DocProjTiny"
diwproj_root = "/mnt/hdd/datasets/documents/diw"
doc3d_root = "/mnt/hdd/datasets/documents/Doc3D"

train_transforms = transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)])
val_transforms = transforms.Compose([RescaleT(288), ToTensorLab(flag=0)])

bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    #    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
    #        loss5.data.item(),
    #        loss6.data.item()))

    return loss0, loss


# define the LightningModule
class WrapperModel(pl.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()

        if model_name == 'u2net':
            self.model = U2NET(3, 1)
        elif model_name == 'u2netp':
            self.model = U2NETP(3, 1)

    def on_fit_start(self):
        self.tb_log = self.logger.experiment

    def training_step(self, batch, batch_idx):

        inputs, labels = batch['image'].float(), batch['label'].float()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = self.model(inputs)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['image'].float(), batch['label'].float()
        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

        if batch_idx == 0 and self.global_rank == 0:
            pred = d1[:, 0, :, :]
            pred_norm = normPRED(pred)
            self.tb_log.add_images("val/images", inputs, global_step=self.global_step)
            self.tb_log.add_images("val/preds_norm", pred_norm.unsqueeze(1), global_step=self.global_step)
            self.tb_log.add_images("val/preds", pred.unsqueeze(1), global_step=self.global_step)
            self.tb_log.add_images("val/masks", labels, global_step=self.global_step)

        self.log_dict({"val/loss": loss.item(),
                       "val/loss_tar": loss2.item(),
                       }, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('medium')
    precision = "32"
    num_gpus = 1

    batch_size_train = 12 * num_gpus

    model_ckpt_args = {
        "dirpath": "output",
        "filename": "{epoch:d}",
        "monitor": "val/loss",
        "mode": "min",
        "save_top_k": 1,
    }
    trainer = pl.Trainer(max_epochs=100,
                         accelerator="cuda", devices=num_gpus,
                         # strategy="ddp",
                         precision=precision,
                         logger=TensorBoardLogger(""),
                         callbacks=[ModelCheckpoint(**model_ckpt_args)],
                         num_sanity_val_steps=1,
                         )

    train_dataset = ConcatDataset([DocProjDataset(root_dir=docproj_root, split="train", transform=train_transforms),
                                   DIWDataset(root_dir=diwproj_root, split="train", transform=train_transforms),
                                   Doc3DDataset(root_dir=doc3d_root, split="train", transform=train_transforms),
                                   ])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4,
                                  pin_memory=True)

    val_dataset = ConcatDataset([DocProjDataset(root_dir=docproj_root, split="val", transform=val_transforms),
                                 DIWDataset(root_dir=diwproj_root, split="val", transform=val_transforms),
                                 Doc3DDataset(root_dir=doc3d_root, split="val", transform=val_transforms),
                                 ])
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    u2net = WrapperModel("u2netp")
    trainer.fit(model=u2net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
