import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def image_pair_key(filename):
    return filename.replace('_image_', '_slice_')


def get_matched_files(rgb_dir):
    inp_dir = os.path.join(rgb_dir, 'input')
    tar_dir = os.path.join(rgb_dir, 'target')
    inp_files = {x: os.path.join(inp_dir, x) for x in sorted(os.listdir(inp_dir)) if is_image_file(x)}
    tar_files = {x: os.path.join(tar_dir, x) for x in sorted(os.listdir(tar_dir)) if is_image_file(x)}
    common_files = sorted(set(inp_files) & set(tar_files))

    if not common_files:
        inp_files = {image_pair_key(x): path for x, path in inp_files.items()}
        tar_files = {image_pair_key(x): path for x, path in tar_files.items()}
        common_files = sorted(set(inp_files) & set(tar_files))

    if not common_files:
        raise RuntimeError(f'No matching input/target image filenames found in {rgb_dir}')

    return [(inp_files[x], tar_files[x]) for x in common_files]


def pad_to_patch_size(img, patch_size):
    w, h = img.size
    padw = max(patch_size - w, 0)
    padh = max(patch_size - h, 0)
    if padw != 0 or padh != 0:
        img = TF.pad(img, (0, 0, padw, padh), padding_mode='reflect')
    return img


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        self.image_pairs = get_matched_files(rgb_dir)
        self.inp_filenames = [inp_path for inp_path, _ in self.image_pairs]
        self.tar_filenames = [tar_path for _, tar_path in self.image_pairs]

        self.img_options = img_options
        self.sizex = len(self.image_pairs)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        #print(os.path.basename(inp_path),'<-->', os.path.basename(tar_path))
        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Reflect pad each image independently in case it is smaller than patch_size.
        inp_img = pad_to_patch_size(inp_img, ps)
        tar_img = pad_to_patch_size(tar_img, ps)
        #print('Min | Max: ',inp_img.getextrema())
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh = min(inp_img.shape[1], tar_img.shape[1])
        ww = min(inp_img.shape[2], tar_img.shape[2])

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        self.image_pairs = get_matched_files(rgb_dir)
        self.inp_filenames = [inp_path for inp_path, _ in self.image_pairs]
        self.tar_filenames = [tar_path for _, tar_path in self.image_pairs]

        self.img_options = img_options
        self.sizex = len(self.image_pairs)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Validate on center crop
        if self.ps is not None:
            inp_img = pad_to_patch_size(inp_img, ps)
            tar_img = pad_to_patch_size(tar_img, ps)
            inp_img = TF.center_crop(inp_img, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')

        inp = TF.to_tensor(inp)
        return inp, filename
