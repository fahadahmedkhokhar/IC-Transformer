import os
from dataset_RGB import BraTSNiftiSliceDataset, DataLoaderTrain, DataLoaderVal, DataLoaderTest, get_dataset_format


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    if get_dataset_format(rgb_dir, img_options) == 'brats_nifti':
        img_options = dict(img_options or {})
        img_options['split'] = 'train'
        return BraTSNiftiSliceDataset(rgb_dir, img_options, is_train=True)
    return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    if get_dataset_format(rgb_dir, img_options) == 'brats_nifti':
        img_options = dict(img_options or {})
        img_options['split'] = img_options.get('split', 'val')
        return BraTSNiftiSliceDataset(rgb_dir, img_options, is_train=False)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    if get_dataset_format(rgb_dir, img_options) == 'brats_nifti':
        img_options = dict(img_options or {})
        img_options['split'] = 'test'
        return BraTSNiftiSliceDataset(rgb_dir, img_options, is_train=False)
    return DataLoaderTest(rgb_dir, img_options)
