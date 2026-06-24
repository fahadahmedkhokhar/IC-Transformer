import os
import torch
import yaml
from torch.utils.data import DataLoader
from model.SUNet import SUNet_model
from data_RGB import get_validation_data
import utils

# Load configuration
with open('training.yaml', 'r') as config_file:
    opt = yaml.safe_load(config_file)

Train = opt['TRAINING']
OPT = opt.get('OPT', opt.get('OPTIM', {}))
Testing = opt.get('TESTING', {})
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
model_path = Testing.get('MODEL_PATH')
if not model_path or not os.path.exists(model_path):
    model_path = utils.get_last_path(model_dir, '_latest.pth')
print(f"Loading checkpoint: {model_path}")
utils.load_checkpoint(model, model_path)
model.eval()

# Load test data with paired targets for metric calculation
test_dir = Testing.get('TEST_DIR', Train.get('TEST_DIR'))
test_options = dict(DatasetOpt)
test_options['patch_size'] = Train['TEST_PS']
test_options['split'] = 'test'
test_dataset = get_validation_data(test_dir, test_options)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
print(f"Testing images: {len(test_dataset)}")

# Run inference and evaluation
psnr_vals = []
ssim_vals = []

print("Running inference on Testing set...")
with torch.no_grad():
    for data_val in test_loader:
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        output = model(input_)
        for res, tar in zip(output, target):
            psnr_vals.append(utils.torchPSNR(tar, res))
            ssim_vals.append(utils.torchSSIM(tar.unsqueeze(0), res.unsqueeze(0)))

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
