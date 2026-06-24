import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.SUNet import SUNet_model
from data_RGB import get_validation_data
import utils

# Load configuration
with open('training.yaml', 'r') as config_file:
    opt = yaml.safe_load(config_file)

Train = opt['TRAINING']
OPT = opt.get('OPT', opt.get('OPTIM', {}))
DatasetOpt = opt.get('DATASET', {})
Testing = opt.get('TESTING', {})
mode = opt['MODEL']['MODE']
input_modalities = DatasetOpt.get('INPUT_MODALITIES', ['t1n', 't2w', 't2f'])

# GPU setup
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

device_ids = [i for i in range(torch.cuda.device_count())]

# Load model
print("Loading model...")
model = SUNet_model(opt).cuda()
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

# Load checkpoint
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
model_path = Testing.get('MODEL_PATH')
if not model_path or not os.path.exists(model_path):
    model_path = utils.get_last_path(model_dir, '_latest.pth')
print(f"Loading checkpoint: {model_path}")
utils.load_checkpoint(model, model_path)
model.eval()

# Load test data
test_dir = Testing.get('TEST_DIR', Train.get('TEST_DIR'))
test_options = dict(DatasetOpt)
test_options['patch_size'] = Train['TEST_PS']
test_options['split'] = 'test'
test_dataset = get_validation_data(test_dir, test_options)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

# Create results directory
results_dir = Testing.get('RESULT_DIR', os.path.join(Train['SAVE_DIR'], mode, 'test_visuals'))
max_visuals = Testing.get('MAX_VISUALS')
os.makedirs(results_dir, exist_ok=True)

# Inference loop
psnr_vals = []
ssim_vals = []

def to_2d_image(tensor):
    array = tensor.detach().cpu().clamp(0, 1).numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        return np.transpose(array, (1, 2, 0))
    return np.squeeze(array)


print("Running inference on Testing set...")
with torch.no_grad():
    for i, data_val in enumerate(test_loader):
        if max_visuals is not None and i >= int(max_visuals):
            break

        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        filenames = data_val[2] if len(data_val) > 2 else [f"image_{i:04d}.png"]

        output = model(input_)

        for j, (res, tar, fname) in enumerate(zip(output, target, filenames)):
            psnr_vals.append(utils.torchPSNR(tar, res))
            ssim_vals.append(utils.torchSSIM(tar.unsqueeze(0), res.unsqueeze(0)))

            input_np = input_[j].detach().cpu().clamp(0, 1).numpy()
            target_img = to_2d_image(tar)
            output_img = to_2d_image(res)
            diff_img = np.abs(target_img - output_img)

            fig, axs = plt.subplots(2, 3, figsize=(16, 10))
            axs = axs.ravel()

            for channel_idx in range(min(3, input_np.shape[0])):
                modality = input_modalities[channel_idx] if channel_idx < len(input_modalities) else f"Input {channel_idx + 1}"
                axs[channel_idx].imshow(input_np[channel_idx], cmap='gray', vmin=0, vmax=1)
                axs[channel_idx].set_title(modality.upper())
                axs[channel_idx].axis("off")

            axs[3].imshow(target_img, cmap='gray', vmin=0, vmax=1)
            axs[3].set_title("GT T1CE")
            axs[3].axis("off")

            axs[4].imshow(output_img, cmap='gray', vmin=0, vmax=1)
            axs[4].set_title("Reconstructed T1CE")
            axs[4].axis("off")

            diff_plot = axs[5].imshow(diff_img, cmap='inferno', vmin=0, vmax=max(float(diff_img.max()), 1e-6))
            axs[5].set_title("Absolute Difference")
            axs[5].axis("off")
            fig.colorbar(diff_plot, ax=axs[5], fraction=0.046, pad=0.04)

            plt.suptitle(f"File: {fname}", fontsize=14)
            plt.tight_layout()

            save_path = os.path.join(results_dir, f"{os.path.splitext(fname)[0]}_inputs_gt_recon_diff.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

# Compute and print average metrics
avg_psnr = torch.stack(psnr_vals).mean().item()
avg_ssim = torch.stack(ssim_vals).mean().item()

print("\n====== Testing Results ======")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Reconstructed images saved to: {results_dir}")
