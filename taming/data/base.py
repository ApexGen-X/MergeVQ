import bisect
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from taming.data.utils import (make_classification_train_transform, make_randomcrop_train_transform,
    make_classification_eval_transform, make_randaugment_train_transform, make_contrastive_train_transform,
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    """Simple ImageNet Dataset for Generation"""
    def __init__(self, paths, size=None, is_training=False, labels=None):
        self.size = size
        self.is_training = is_training

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            if self.is_training:
                self.preprocessor = make_randomcrop_train_transform(
                    resize_size=self.size, crop_size=self.size,
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            else:
                self.preprocessor = make_classification_eval_transform(
                    resize_size=self.size, crop_size=self.size, interpolation=T.InterpolationMode.BILINEAR,
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.preprocessor(image)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


class MultiImagePaths(Dataset):
    """Multiple Image Dataset for Contrastive Learning"""
    def __init__(self, paths, size=None, is_training=False, labels=None, augmentation_type='default'):
        self.size = size
        self.is_training = is_training
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        if isinstance(augmentation_type, str):
            self.augmentation_type = [augmentation_type] * len(self.size)
        elif isinstance(augmentation_type, list):
            self.augmentation_type = augmentation_type if len(augmentation_type) >= len(self.size) \
                else ['default'] * (len(self.size)-len(augmentation_type)) + augmentation_type

        assert len(self.size) >= 2
        if self.is_training:
            self.preprocessor_d = make_randomcrop_train_transform(  # default (inception norm)
                resize_size=self.size[0], crop_size=self.size[0],
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            # teacher (weak aug with default norm)
            if self.augmentation_type[1] != 'rand_augment':
                if self.augmentation_type[1] == 'randomcrop':
                    self.preprocessor_w = make_randomcrop_train_transform(
                        resize_size=self.size[1], crop_size=self.size[1],
                        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                else:  # default
                    self.preprocessor_w = make_classification_train_transform(
                        crop_size=self.size[1], crop_scale=[0.2, 1],
                        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            else:
                self.preprocessor_w = None
            # student (strong aug with inception norm)
            if self.augmentation_type[-1] == 'contrastive':
                self.preprocessor_s = make_contrastive_train_transform(
                    crop_size=self.size[-1],
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            elif self.augmentation_type[-1] == 'rand_augment':
                self.preprocessor_s = make_randaugment_train_transform(
                    crop_size=self.size[-1], auto_augment='rand-m7-mstd0.5-inc1',
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            else:  # default
                self.preprocessor_s = make_classification_train_transform(
                    crop_size=self.size[-1],
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        else:
            self.preprocessor_d = make_classification_eval_transform(  # default
                resize_size=self.size[0], crop_size=self.size[0], interpolation=T.InterpolationMode.BILINEAR,
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            self.preprocessor_w = make_classification_eval_transform(  # teacher (weak aug)
                resize_size=self.size[1], crop_size=self.size[1], interpolation=T.InterpolationMode.BILINEAR,
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            self.preprocessor_s = make_classification_eval_transform(  # student (strong aug)
                resize_size=self.size[-1], crop_size=self.size[-1], interpolation=T.InterpolationMode.BILINEAR,
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image_d = self.preprocessor_d(image)
        image_w = self.preprocessor_w(image) \
            if self.preprocessor_w is not None else image_d
        image_s = self.preprocessor_s(image)
        # image_s = image_w
        return (image_d, image_w, image_s)

    def __getitem__(self, i):
        example = dict()
        example["image"], example["image_aug_w"], example["image_aug_s"] = \
            self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
