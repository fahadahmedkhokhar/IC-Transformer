import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
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
mode = opt['MODEL']['MODE']

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
model_path = utils.get_last_path(model_dir, '_latest.pth')
utils.load_checkpoint(model, model_path)
model.eval()

# Load validation data
val_dir = Train['TRAIN_DIR']
val_options = dict(DatasetOpt)
val_options['patch_size'] = Train['TEST_PS']
val_dataset = get_validation_data(val_dir, val_options)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

# Create results directory
results_dir = "/home/users/fahadahmed.khokhar/temp/hamnah/Transformer-Training1/Saved_Reconstructed"
os.makedirs(results_dir, exist_ok=True)

# Inference loop
psnr_vals = []
ssim_vals = []

print("Running inference on Testing set...")
with torch.no_grad():
    for i, data_val in enumerate(val_loader):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        filenames = data_val[2] if len(data_val) > 2 else [f"image_{i:04d}.png"]

        output = model(input_)

        for j, (res, tar, fname) in enumerate(zip(output, target, filenames)):
            psnr_vals.append(utils.torchPSNR(res, tar))
            ssim_vals.append(utils.torchSSIM(res.unsqueeze(0), tar.unsqueeze(0)))

            # Convert tensors to NumPy
            input_np = input_[j].detach().cpu().clamp(0, 1).numpy()
            target_np = tar.detach().cpu().clamp(0, 1).numpy()
            output_np = res.detach().cpu().clamp(0, 1).numpy()

            if input_np.shape[0] == 3:
                input_np = np.transpose(input_np, (1, 2, 0))
            elif input_np.shape[0] == 1:
                input_np = input_np[0]
            if target_np.shape[0] == 3:
                target_np = np.transpose(target_np, (1, 2, 0))
            elif target_np.shape[0] == 1:
                target_np = target_np[0]
            if output_np.shape[0] == 3:
                output_np = np.transpose(output_np, (1, 2, 0))
            elif output_np.shape[0] == 1:
                output_np = output_np[0]

            input_img = (input_np * 255).astype(np.uint8)
            target_img = (target_np * 255).astype(np.uint8)
            output_img = (output_np * 255).astype(np.uint8)

            # === Save the matplotlib image with titles ===
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(input_img)
            axs[0].set_title("Input Image")
            axs[0].axis("off")

            axs[1].imshow(target_img)
            axs[1].set_title("Target (Ground Truth)")
            axs[1].axis("off")

            axs[2].imshow(output_img)
            axs[2].set_title("Reconstructed Image")
            axs[2].axis("off")

            plt.suptitle(f"File: {fname}", fontsize=14)
            plt.tight_layout()

            # Save the figure as PNG
            save_path = os.path.join(results_dir, f"{os.path.splitext(fname)[0]}_with_titles.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()
            plt.close()

# Compute and print average metrics
avg_psnr = torch.stack(psnr_vals).mean().item()
avg_ssim = torch.stack(ssim_vals).mean().item()

print("\n====== Testing Results ======")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Reconstructed images saved to: {results_dir}")
