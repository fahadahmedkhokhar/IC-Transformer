#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 02:44:35 2023

@author: nadeem
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import numpy as np
import argparse
from model.SUNet import SUNet_model
import math
from tqdm import tqdm
import yaml

from pathlib import Path
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)


'''parser = argparse.ArgumentParser(description='Image Restoration')
#if not os.path.isdir('results'):
#    os.mkdir('results')
parser.add_argument('--input_dir', default='datasets/3d_as_2d_meanCBF60_data_CBFlow_resize_rotate_6july/val/input/', type=str, help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size')
parser.add_argument('--size', default=256, type=int, help='model image patch size')
parser.add_argument('--stride', default=128, type=int, help='reconstruction stride')
parser.add_argument('--result_dir', default='datasets/results_3d_as_2d_meanCBF60_data_CBFlow_resize_rotate_6july_bestPSNR/val/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='Sampledata_checkpoints/Denoising/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')'''

parser = argparse.ArgumentParser(description='Image Restoration')
#if not os.path.isdir('results'):
#    os.mkdir('results')
parser.add_argument('--input_dir', default='datasets/2D_PCASL_label_meanCBF_5_dec_2023/task_oriented_2D_PCASL_data_2/input/', type=str, help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size')
parser.add_argument('--size', default=256, type=int, help='model image patch size')
parser.add_argument('--stride', default=128, type=int, help='reconstruction stride')
parser.add_argument('--result_dir', default='datasets/results_2D_PCASL_label_meanCBF_5_dec_2023_bestSSIM/task_oriented_2D_PCASL_data_2/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='2D_PCASL_label_meanCBF_5_dec_2023_checkpoints/Denoising/models/model_bestSSIM.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()

if not os.path.exists(args.result_dir):
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
def overlapped_square(timg, kernel=256, stride=128):
    patch_images = []
    b, c, h, w = timg.size()
    # 321, 481
    X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h, w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
    patch = patch.contiguous().view(b, c, -1, kernel, kernel)  # B, C, #patches, K, K
    patch = patch.permute(2, 0, 1, 4, 3)  # patches, B, C, K, K

    for each in range(len(patch)):
        patch_images.append(patch[each])

    return patch_images, mask, X


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

files=natsorted(glob(inp_dir+'*.*'))#                            ('SampleData/Input/*/*.*'))

#%%
if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")


# Load corresponding model architecture and weights
model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')
print(f"Results saving at {out_dir}")
stride = args.stride
model_img = args.size

for file_ in tqdm(files):
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    with torch.no_grad():
        # pad to multiple of 256
        square_input_, mask, max_wh = overlapped_square(input_.cuda(), kernel=model_img, stride=stride)
        output_patch = torch.zeros(square_input_[0].shape).type_as(square_input_[0])
        for i, data in enumerate(square_input_):
            restored = model(square_input_[i])
            if i == 0:
                output_patch += restored
            else:
                output_patch = torch.cat([output_patch, restored], dim=0)

        B, C, PH, PW = output_patch.shape
        weight = torch.ones(B, C, PH, PH).type_as(output_patch)  # weight_mask

        patch = output_patch.contiguous().view(B, C, -1, model_img*model_img)
        patch = patch.permute(2, 1, 3, 0)  # B, C, K*K, #patches
        patch = patch.contiguous().view(1, C*model_img*model_img, -1)

        weight_mask = weight.contiguous().view(B, C, -1, model_img * model_img)
        weight_mask = weight_mask.permute(2, 1, 3, 0)  # B, C, K*K, #patches
        weight_mask = weight_mask.contiguous().view(1, C * model_img * model_img, -1)

        restored = F.fold(patch, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
        we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
        restored /= we_mk

        restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)
        restored = torch.clamp(restored, 0, 1)

    
    font = cv2.FONT_HERSHEY_COMPLEX
    
    
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])
    
    
    #restored=cv2.putText(restored,'denoised',(0,30),font,1,(255,0,0),2)
    
    
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    #print("restored: ",restored.shape)
    #print("square_input_[i] :",square_input_[i].shape)
    #print("input_ :",input_.shape)
    img=np.asarray(img)
    
    white_line = np.ones((img.shape[0], 10,3))
    #ff=np.concatenate([img, restored],axis=1)
    
    dd=img[0:40,0:40,:3]
    if(np.mean(dd)>127):
        co=0
    else:
        co=255
        
    #cv2.putText(ff,'Noisy',(0,10),font,0.3,(255,0,0),1)
    #pos=int(ff.shape[1]/2)
    #cv2.putText(ff,'Denoised',(pos,10),font,0.3,(0,0,255),1)
    #print(ff.shape)
   
    #nam_gen = '/'.join(nam_gen.split('\\'))
   # s=os.path.dirname(os.path.join(os.getcwd(),nam_gen))
    
    
    if not os.path.exists(os.path.join(out_dir,'input')):
        os.mkdir(os.path.join(out_dir,'input'))
        
    if not os.path.exists(os.path.join(out_dir,'target')):
        os.mkdir(os.path.join(out_dir,'target'))
        
    #if not os.path.exists(s):
    #   Path(s).mkdir(parents=True, exist_ok=True)
    
    nam_gen_input=os.path.join(os.getcwd(),out_dir,'input',f+'.png')
    nam_gen_target=os.path.join(os.getcwd(),out_dir,'target',f+'.png')
    
    
    #print(s)
    #print(nam_gen)
   
    save_img(nam_gen_input, img)
    save_img(nam_gen_target, restored)

