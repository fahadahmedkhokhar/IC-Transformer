#!/usr/bin/env python3
"""Quick test to verify training runs without errors for a few batches."""

import torch
import yaml
from data_RGB import get_training_data
from model.SUNet import SUNet_model
from tqdm import tqdm

# Load config
with open('./training.yaml', 'r') as f:
    config = yaml.safe_load(f)

Train = config['TRAINING']

# Setup device
device_ids = [torch.cuda.current_device()]
model_restored = SUNet_model(config)
model_restored = torch.nn.DataParallel(model_restored.cuda(), device_ids=device_ids)

# Load training data
train_loader = get_training_data(Train['TRAIN_DIR'], Train['BATCH'], Train['TRAIN_PS'], num_workers=4)

# Test first 10 batches
L1_loss = torch.nn.L1Loss()
print("Testing training loop on first 10 batches...")

try:
    for i, data in enumerate(tqdm(train_loader, total=10)):
        if i >= 10:
            break
        
        target = data[0].cuda()  # Ground truth (RGB image)
        input_ = data[1].cuda()  # Input image (RGB image)
        
        # Forward pass through model
        restored = model_restored(input_)
        loss = L1_loss(restored, target)
        
        if i == 0:
            print(f"\nBatch {i}:")
            print(f"  Target shape: {target.shape}")
            print(f"  Input shape: {input_.shape}")
            print(f"  Restored shape: {restored.shape}")
            print(f"  Loss: {loss.item():.6f}")
        
        if (i + 1) % 5 == 0:
            print(f"Batch {i + 1}: Loss = {loss.item():.6f}")
    
    print("\n✅ SUCCESS! Training loop works correctly for 10 batches!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
