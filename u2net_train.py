from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim as optim
import lightning as L
from lightning_fabric.loggers import TensorBoardLogger

from data_loader import DocProjDataset, RescaleT, RandomCrop, ToTensorLab

from model import U2NET, U2NETP
from u2net_test import normPRED

dataset_root = "/mnt/hdd/datasets/documents/DocProjTiny"

# ------- 1. define loss function --------

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
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(),
        loss6.data.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------
dataset_root = "/mnt/hdd/datasets/documents/DocProjTiny"

model_name = 'u2netp'  # 'u2net'

image_ext = '.jpg'
label_ext = '.png'

model_dir = Path('saved_models') / model_name
model_dir.mkdir(exist_ok=True, parents=True)

epoch_num = 100000
batch_size_train = 12
batch_size_val = 1

train_dataset = DocProjDataset(
    root_dir=dataset_root,
    split="train",
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
val_dataset = DocProjDataset(
    root_dir=dataset_root,
    split="val",
    transform=transforms.Compose([
        RescaleT(288),
        ToTensorLab(flag=0)]))

print("---")
print("train images: ", len(train_dataset))
print("train labels: ", len(val_dataset))
print("---")

train_num = len(train_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=4)

# ------- 3. define model --------
# define the net
if (model_name == 'u2net'):
    net = U2NET(3, 1)
elif (model_name == 'u2netp'):
    net = U2NETP(3, 1)

fabric = L.Fabric(accelerator="cuda", devices=1, loggers=TensorBoardLogger(""))
fabric.launch()
tb_logger: SummaryWriter = fabric.logger.experiment
# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
net, optimizer = fabric.setup(net, optimizer)
val_dataloader, train_dataloader = fabric.setup_dataloaders(val_dataloader, train_dataloader)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2  # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(train_dataloader):
        ite_num += 1
        ite_num4val += 1

        inputs, labels = data['image'].float(), data['label'].float()

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

        fabric.backward(loss)
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()
        fabric.log_dict(
            {
                "epoch": epoch,
                "train/loss": running_loss / ite_num4val,
                "train/loss_tar": running_tar_loss / ite_num4val,
            }, step=epoch * len(train_dataloader) + i
        )

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
            running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(),
                       model_dir / f"{model_name}_bce_itr_{ite_num}_train_{running_loss / ite_num4val:.3f}_tar_{running_tar_loss / ite_num4val:.3f}.pth")
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
    print(f"Validation {epoch}")
    val_losses = 0.0
    val_losses_tar = 0.0

    for i, data in enumerate(val_dataloader):
        inputs, labels = data['image'].float(), data['label'].float()
        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

        if i == 0:
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)
            tb_logger.add_image("val/image", inputs[0], global_step=epoch)
            tb_logger.add_image("val/pred", pred[0][None], global_step=epoch)
            tb_logger.add_image("val/mask", labels[0], global_step=epoch)

            tb_logger.add_images("val/images", inputs, global_step=epoch)
            tb_logger.add_images("val/preds", pred.unsqueeze(1), global_step=epoch)
            tb_logger.add_images("val/masks", labels, global_step=epoch)

        # # print statistics
        val_losses += loss.data.item()
        val_losses_tar += loss2.data.item()

    fabric.log_dict({"val/loss": val_losses / len(val_dataloader),
                     "val/loss_tar": val_losses_tar / len(val_dataloader),
                     }, step=epoch)
