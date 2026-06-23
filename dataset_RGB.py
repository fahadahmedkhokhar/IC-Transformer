import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


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


def strip_nii_ext(filename):
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def is_nifti_file(filename):
    return filename.endswith('.nii') or filename.endswith('.nii.gz')


def find_modality_file(case_dir, modality):
    matches = [
        os.path.join(case_dir, name)
        for name in os.listdir(case_dir)
        if is_nifti_file(name) and strip_nii_ext(name).lower().endswith(f'-{modality.lower()}')
    ]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one *-{modality}.nii(.gz) file in {case_dir}, found {len(matches)}')
    return matches[0]


def normalize_volume(volume, eps=1e-8):
    volume = volume.astype(np.float32)
    mask = volume != 0
    if np.any(mask):
        values = volume[mask]
        v_min = values.min()
        v_max = values.max()
    else:
        v_min = volume.min()
        v_max = volume.max()

    if v_max - v_min < eps:
        return np.zeros_like(volume, dtype=np.float32)

    volume = (volume - v_min) / (v_max - v_min)
    volume = np.clip(volume, 0.0, 1.0)
    return volume.astype(np.float32)


def load_nifti(path):
    if nib is None:
        raise ImportError('nibabel is required for NIfTI datasets. Install it with: pip install nibabel')
    return nib.load(path).get_fdata(dtype=np.float32)


def get_slice(volume, index, axis):
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError(f'Unsupported slice axis: {axis}')


def pad_tensor_to_patch_size(img, patch_size):
    _, h, w = img.shape
    pad_h = max(patch_size - h, 0)
    pad_w = max(patch_size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return img

    pad_mode = 'reflect' if h > 1 and w > 1 else 'replicate'
    return F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode=pad_mode).squeeze(0)


def get_dataset_format(data_dir, img_options):
    options = img_options or {}
    requested = get_option(options, 'data_format', get_option(options, 'format', 'auto'))
    if requested != 'auto':
        return requested
    if os.path.isdir(os.path.join(data_dir, 'input')) and os.path.isdir(os.path.join(data_dir, 'target')):
        return 'rgb'
    return 'brats_nifti'


def get_option(options, key, default=None):
    if key in options:
        return options[key]
    upper_key = key.upper()
    if upper_key in options:
        return options[upper_key]
    return default


class BraTSNiftiSliceDataset(Dataset):
    def __init__(self, root_dir, img_options=None, is_train=True):
        super(BraTSNiftiSliceDataset, self).__init__()
        self.root_dir = root_dir
        self.img_options = img_options or {}
        self.is_train = is_train
        self.split = get_option(self.img_options, 'split', 'train' if is_train else 'val')
        self.ps = get_option(self.img_options, 'patch_size')
        self.input_modalities = get_option(self.img_options, 'input_modalities', ['t1n', 't2w', 't2f'])
        self.target_modality = get_option(self.img_options, 'target_modality', 't1c')
        self.slice_axis = int(get_option(self.img_options, 'slice_axis', 2))
        self.skip_empty_target = bool(get_option(self.img_options, 'skip_empty_target', True))
        self.min_target_pixels = int(get_option(self.img_options, 'min_target_pixels', 1))
        self.cache_volumes = bool(get_option(self.img_options, 'cache_volumes', True))
        self.split_seed = int(get_option(self.img_options, 'split_seed', 1234))
        self.train_ratio = float(get_option(self.img_options, 'train_ratio', 0.7))
        self.val_ratio = float(get_option(self.img_options, 'val_ratio', 0.15))
        self.test_ratio = float(get_option(self.img_options, 'test_ratio', 0.15))
        self.volume_cache = {}

        self.cases = self._index_cases()
        self.samples = self._index_slices()
        if not self.samples:
            raise RuntimeError(f'No usable NIfTI slices found in {root_dir}')

    def _index_cases(self):
        cases = []
        for name in sorted(os.listdir(self.root_dir)):
            case_dir = os.path.join(self.root_dir, name)
            if not os.path.isdir(case_dir):
                continue
            paths = {modality: find_modality_file(case_dir, modality) for modality in self.input_modalities}
            paths[self.target_modality] = find_modality_file(case_dir, self.target_modality)
            cases.append({'name': name, 'paths': paths})
        if not cases:
            raise RuntimeError(f'No case folders found in {self.root_dir}')
        return self._select_split_cases(cases)

    def _select_split_cases(self, cases):
        if self.split in ('all', None):
            return cases

        rng = random.Random(self.split_seed)
        shuffled = list(cases)
        rng.shuffle(shuffled)

        total = len(shuffled)
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if ratio_sum <= 0:
            raise ValueError('DATASET split ratios must sum to a positive value')

        train_ratio = self.train_ratio / ratio_sum
        val_ratio = self.val_ratio / ratio_sum

        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        if total >= 3:
            n_train = min(max(1, n_train), total - 2)
            n_val = min(max(1, n_val), total - n_train - 1)
        elif total == 2:
            n_train = 1
            n_val = 1
        elif total == 1:
            n_train = 1
            n_val = 0

        split_ranges = {
            'train': shuffled[:n_train],
            'val': shuffled[n_train:n_train + n_val],
            'validation': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:],
        }

        selected = split_ranges.get(self.split)
        if selected is None:
            raise ValueError(f'Unsupported DATASET split: {self.split}')
        if not selected:
            raise RuntimeError(f'No cases selected for split "{self.split}" from {self.root_dir}')
        return sorted(selected, key=lambda case: case['name'])

    def _index_slices(self):
        samples = []
        for case_idx, case in enumerate(self.cases):
            target = load_nifti(case['paths'][self.target_modality])
            num_slices = target.shape[self.slice_axis]
            for slice_idx in range(num_slices):
                target_slice = get_slice(target, slice_idx, self.slice_axis)
                if self.skip_empty_target and np.count_nonzero(target_slice) < self.min_target_pixels:
                    continue
                samples.append((case_idx, slice_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_case_volume(self, case_idx, modality):
        cache_key = (case_idx, modality)
        if self.cache_volumes and cache_key in self.volume_cache:
            return self.volume_cache[cache_key]

        case = self.cases[case_idx]
        volume = normalize_volume(load_nifti(case['paths'][modality]))
        if self.cache_volumes:
            self.volume_cache[cache_key] = volume
        return volume

    def __getitem__(self, index):
        case_idx, slice_idx = self.samples[index]
        case = self.cases[case_idx]

        input_slices = []
        for modality in self.input_modalities:
            volume = self._load_case_volume(case_idx, modality)
            input_slices.append(get_slice(volume, slice_idx, self.slice_axis))

        target_volume = self._load_case_volume(case_idx, self.target_modality)
        target_slice = get_slice(target_volume, slice_idx, self.slice_axis)

        inp_img = torch.from_numpy(np.stack(input_slices, axis=0)).float()
        tar_img = torch.from_numpy(target_slice[None, ...]).float()

        if self.ps is not None:
            inp_img = pad_tensor_to_patch_size(inp_img, self.ps)
            tar_img = pad_tensor_to_patch_size(tar_img, self.ps)

            if self.is_train:
                hh = min(inp_img.shape[1], tar_img.shape[1])
                ww = min(inp_img.shape[2], tar_img.shape[2])
                rr = random.randint(0, hh - self.ps)
                cc = random.randint(0, ww - self.ps)
            else:
                hh = min(inp_img.shape[1], tar_img.shape[1])
                ww = min(inp_img.shape[2], tar_img.shape[2])
                rr = max((hh - self.ps) // 2, 0)
                cc = max((ww - self.ps) // 2, 0)

            inp_img = inp_img[:, rr:rr + self.ps, cc:cc + self.ps]
            tar_img = tar_img[:, rr:rr + self.ps, cc:cc + self.ps]

        if self.is_train:
            aug = random.randint(0, 8)
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

        filename = f'{case["name"]}_slice_{slice_idx:03d}'
        return tar_img, inp_img, filename


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
