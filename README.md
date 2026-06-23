# IC-Transformer

Transformer-based image restoration training code for paired MRI/RGB-style image denoising experiments. The project trains a SUNet/Swin-Transformer style model using paired `input` and `target` images and reports PSNR and SSIM during validation/testing.

## Project Structure

```text
.
├── train.py                  # Main training script
├── test.py                   # Evaluation script with PSNR/SSIM and 95% CI
├── training.yaml             # Model, optimizer, dataset, and checkpoint config
├── data_RGB.py               # Dataset factory functions
├── dataset_RGB.py            # Paired input/target image dataset loaders
├── model/                    # SUNet model definitions
├── utils/                    # Metrics, checkpoint, image, and directory helpers
├── warmup_scheduler/         # Warmup + cosine learning rate scheduler
└── requirements.txt          # Python dependencies
```

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended
- PyTorch with CUDA support

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails, install the correct CUDA build for your system from the official PyTorch instructions, then install the remaining packages from `requirements.txt`.

## Dataset Format

Each split should contain paired images in `input` and `target` folders:

```text
Dataset/
├── train/
│   ├── input/
│   └── target/
├── val/
│   ├── input/
│   └── target/
└── test/
    ├── input/
    └── target/
```

Input and target filenames should match. The loader also supports matching names where `_image_` in the input filename corresponds to `_slice_` in the target filename.

## Configuration

Edit `training.yaml` before running:

```yaml
TRAINING:
  TRAIN_DIR: path/to/Dataset/train
  VAL_DIR: path/to/Dataset/val
  TEST_DIR: path/to/Dataset/test
  SAVE_DIR: path/to/output/checkpoints

TESTING:
  MODEL_PATH: path/to/model_bestSSIM.pth
  TEST_DIR: path/to/Dataset/test
```

Important options:

- `GPU`: GPU IDs to use.
- `SWINUNET`: model architecture parameters.
- `OPT.BATCH`: training batch size.
- `OPT.EPOCHS`: number of epochs.
- `OPT.LR_INITIAL`: initial learning rate.
- `TRAINING.RESUME`: set to `False` for a fresh run, or `True` to resume from the latest checkpoint.

## Training

Run:

```bash
python train.py
```

The script saves checkpoints under:

```text
<SAVE_DIR>/Denoising/models/
```

Generated files include:

- `model_latest.pth`
- `model_bestPSNR.pth`
- `model_bestSSIM.pth`

TensorBoard logs are saved under:

```text
<SAVE_DIR>/Denoising/log/
```

## Testing

Run:

```bash
python test.py
```

The test script reports:

- Average PSNR
- Average SSIM
- PSNR mean +/- standard deviation
- SSIM mean +/- standard deviation
- PSNR mean +/- 95% confidence interval
- SSIM mean +/- 95% confidence interval

## Notes

- Large checkpoint files, TensorBoard logs, cache files, and local editor settings are excluded from Git using `.gitignore`.
- Update the absolute Windows paths in `training.yaml` when running on another machine.
- The code currently expects CUDA and calls `.cuda()` in the training/testing scripts.
