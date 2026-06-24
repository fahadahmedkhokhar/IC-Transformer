import os
import torch
import yaml
import time
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import network_parameters
import utils
from data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from model.SUNet import SUNet_model

# ----------------Seeds ------------------
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# ------------------ Load Config ------------------
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPT']
DatasetOpt = opt.get('DATASET', {})

# ------------------ Build Model ------------------
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# ------------------ GPU Settings ------------------
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if len(device_ids) > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

# ------------------ Logging ------------------
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

# ------------------ Optimizer & Scheduler ------------------
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs, eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('Resuming Training with learning rate:', new_lr)

# ------------------ Loss Function ------------------
L1_loss = nn.L1Loss()

# ------------------ Data Loaders ------------------
print('==> Loading datasets')
train_options = dict(DatasetOpt)
train_options['patch_size'] = Train['TRAIN_PS']
val_options = dict(DatasetOpt)
val_options['patch_size'] = Train['VAL_PS']
train_dataset = get_training_data(train_dir, train_options)
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'], shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(val_dir, val_options)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

print(f'Training details:\nModel parameters: {p_number}\nStart/End epochs: {start_epoch}~{OPT["EPOCHS"]}')

# ------------------ Training Loop ------------------
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    model_restored.train()

    for i, data in enumerate(tqdm(train_loader), 0):
        target = data[0].cuda()  # Ground truth T1CE, shape: [B, 1, H, W]
        input_ = data[1].cuda()  # Input T1/T2/FLAIR, shape: [B, 3, H, W]

        restored = model_restored(input_)
        loss = L1_loss(restored, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if epoch == start_epoch and i == 0:
            print(
                "First batch ranges | "
                f"input: [{input_.min().item():.4f}, {input_.max().item():.4f}] | "
                f"target: [{target.min().item():.4f}, {target.max().item():.4f}] | "
                f"output: [{restored.min().item():.4f}, {restored.max().item():.4f}]"
            )

    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()  # Ground truth
            input_ = data_val[1].cuda()  # Input image

            with torch.no_grad():
                restored = model_restored(input_)

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(tar, res))
                ssim_val_rgb.append(utils.torchSSIM(tar.unsqueeze(0), res.unsqueeze(0)))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_bestPSNR.pth"))

        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_bestSSIM.pth"))

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)

    scheduler.step()
    avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
    print(
        "Epoch: {}\tTime: {:.2f}s\tAvg Loss: {:.6f}\tTotal Loss: {:.4f}".format(
            epoch, time.time() - epoch_start_time, avg_epoch_loss, epoch_loss
        )
    )

    torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_latest.pth"))
    writer.add_scalar('train/loss', avg_epoch_loss, epoch)
    writer.add_scalar('train/total_loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

writer.close()
print('Total training time: {:.1f} hours'.format((time.time() - total_start_time) / 3600))
