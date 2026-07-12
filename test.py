import os
import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from model.SUNet import SUNet_model
from data_RGB import get_validation_data
import utils


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


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='training.yaml', help='Path to experiment YAML config')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r', encoding='utf-8-sig') as config_file:
    opt = strip_bom_keys(yaml.safe_load(config_file))

Train = opt['TRAINING']
OPT = opt.get('OPT', opt.get('OPTIM', {}))
Testing = opt.get('TESTING', {})
mode = opt['MODEL']['MODE']
POCS = opt.get('POCS', {})
pocs_enabled = bool(POCS.get('ENABLE', False))
pocs_iterations = int(POCS.get('ITERATIONS', 1))
pocs_acceleration = POCS.get('ACCELERATION', None)
if pocs_acceleration is None:
    pocs_acceleration = parse_acceleration_from_path(test_dir if 'test_dir' in locals() else Train.get('TEST_DIR', Train['TRAIN_DIR']))
if pocs_acceleration is not None:
    pocs_acceleration = int(pocs_acceleration)

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

# Load test data with paired targets for metric calculation
test_dir = Testing.get('TEST_DIR', Train.get('TEST_DIR'))
test_dataset = get_validation_data(test_dir, {'patch_size': Train['TEST_PS']})
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
print(f"Testing images: {len(test_dataset)}")
if pocs_enabled and pocs_acceleration:
    print(f"POCS data consistency: enabled, Cartesian acceleration={pocs_acceleration}x, iterations={pocs_iterations}")
elif pocs_enabled:
    print("POCS data consistency: requested, but no Cartesian acceleration was found. It will be skipped.")
else:
    print("POCS data consistency: disabled")

# Run inference and evaluation
psnr_vals = []
ssim_vals = []

print("Running inference on Testing set...")
with torch.no_grad():
    for data_val in test_loader:
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        output = model(input_)
        if pocs_enabled and pocs_acceleration:
            sampling_mask = cartesian_sampling_mask(output.shape, pocs_acceleration, output.device)
            output = pocs_data_consistency(output, input_, sampling_mask, pocs_iterations)
        for res, tar in zip(output, target):
            psnr_vals.append(utils.torchPSNR(res, tar))
            ssim_vals.append(utils.torchSSIM(res.unsqueeze(0), tar.unsqueeze(0)))

# Compute averages and uncertainty estimates
psnr_tensor = torch.stack(psnr_vals)
ssim_tensor = torch.stack(ssim_vals)
n_samples = psnr_tensor.numel()

avg_psnr = psnr_tensor.mean().item()
avg_ssim = ssim_tensor.mean().item()
std_psnr = psnr_tensor.std(unbiased=True).item()
std_ssim = ssim_tensor.std(unbiased=True).item()
ci95_psnr = 1.96 * std_psnr / (n_samples ** 0.5)
ci95_ssim = 1.96 * std_ssim / (n_samples ** 0.5)

print("\n====== Testing Results ======")
print(f"Number of test images: {n_samples}")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"PSNR mean +/- std: {avg_psnr:.4f} +/- {std_psnr:.4f}")
print(f"SSIM mean +/- std: {avg_ssim:.4f} +/- {std_ssim:.4f}")
print(f"PSNR mean +/- 95% CI: {avg_psnr:.4f} +/- {ci95_psnr:.4f}")
print(f"SSIM mean +/- 95% CI: {avg_ssim:.4f} +/- {ci95_ssim:.4f}")
