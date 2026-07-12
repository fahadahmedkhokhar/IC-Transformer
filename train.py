import os
import argparse
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

def parse_acceleration_from_path(path):
    normalized = path.replace("\\", "/").lower()
    for part in normalized.split("/"):
        if part.startswith("cartesian_") and part.endswith("x"):
            return int(part.split("_")[1].replace("x", ""))
    return None


def cartesian_sampling_mask(shape, acceleration, device):
    _, _, height, width = shape
    columns = max(1, width // acceleration)
    center = width // 2
    start = center - columns // 2
    end = start + columns

    mask = torch.zeros((1, 1, height, width), dtype=torch.bool, device=device)
    mask[:, :, :, start:end] = True
    return mask


def pocs_data_consistency(predicted, measured_input, sampling_mask, iterations=1):
    """Enforce measured k-space samples on the network prediction."""
    if sampling_mask is None or iterations <= 0:
        return predicted

    x = predicted
    measured_kspace = torch.fft.fftshift(
        torch.fft.fft2(measured_input, dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )
    mask = sampling_mask.to(device=predicted.device)

    for _ in range(iterations):
        predicted_kspace = torch.fft.fftshift(
            torch.fft.fft2(x, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )
        corrected_kspace = torch.where(mask, measured_kspace, predicted_kspace)
        x = torch.fft.ifft2(
            torch.fft.ifftshift(corrected_kspace, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ).real

    return x.clamp(0.0, 1.0)


def strip_bom_keys(config):
    return {key.lstrip("\ufeff") if isinstance(key, str) else key: value for key, value in config.items()}

# ----------------Seeds ------------------
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='training.yaml', help='Path to experiment YAML config')
args = parser.parse_args()

# ------------------ Load Config ------------------
with open(args.config, 'r', encoding='utf-8-sig') as config:
    opt = strip_bom_keys(yaml.safe_load(config))
Train = opt['TRAINING']
OPT = opt['OPT']
POCS = opt.get('POCS', {})
pocs_enabled = bool(POCS.get('ENABLE', False))
pocs_iterations = int(POCS.get('ITERATIONS', 1))
pocs_acceleration = POCS.get('ACCELERATION', None)
if pocs_acceleration is None:
    pocs_acceleration = parse_acceleration_from_path(Train['TRAIN_DIR'])
if pocs_acceleration is not None:
    pocs_acceleration = int(pocs_acceleration)

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
    try:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    except IndexError:
        path_chk_rest = None

    if path_chk_rest and os.path.exists(path_chk_rest):
        utils.load_checkpoint(model_restored, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('Resuming Training with learning rate:', new_lr)
    else:
        print('RESUME=True but no checkpoint was found. Starting fresh.')

# ------------------ Loss Function ------------------
L1_loss = nn.L1Loss()

# ------------------ Data Loaders ------------------
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'], shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

print(f'Training details:\nModel parameters: {p_number}\nStart/End epochs: {start_epoch}~{OPT["EPOCHS"]}')
if pocs_enabled and pocs_acceleration:
    print(f'POCS data consistency: enabled, Cartesian acceleration={pocs_acceleration}x, iterations={pocs_iterations}')
elif pocs_enabled:
    print('POCS data consistency: requested, but no Cartesian acceleration was found. It will be skipped.')
else:
    print('POCS data consistency: disabled')

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
        target = data[0].cuda()  # Ground truth (RGB image, shape: [B, 3, H, W])
        input_ = data[1].cuda()  # Input image (RGB image, shape: [B, 3, H, W])

        # Model expects input of shape [B, 3, H, W]
        # Forward pass through the model
        restored = model_restored(input_)
        if pocs_enabled and pocs_acceleration:
            sampling_mask = cartesian_sampling_mask(restored.shape, pocs_acceleration, restored.device)
            restored = pocs_data_consistency(restored, input_, sampling_mask, pocs_iterations)
        loss = L1_loss(restored, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()  # Ground truth
            input_ = data_val[1].cuda()  # Input image

            with torch.no_grad():
                restored = model_restored(input_)
                if pocs_enabled and pocs_acceleration:
                    sampling_mask = cartesian_sampling_mask(restored.shape, pocs_acceleration, restored.device)
                    restored = pocs_data_consistency(restored, input_, sampling_mask, pocs_iterations)

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(res.unsqueeze(0), tar.unsqueeze(0)))

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
    print("Epoch: {}	Time: {:.2f}s	Loss: {:.4f}".format(epoch, time.time() - epoch_start_time, epoch_loss))

    torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_latest.pth"))
    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

writer.close()
print('Total training time: {:.1f} hours'.format((time.time() - total_start_time) / 3600))
