# -*- encoding: utf-8 -*-
import random
import cv2

import torch.utils.data as data
import astra.data.utils as utils


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = utils.random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = utils.random_rotate_around_z_axis(
        list_images,
        list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
        list_boder_value=(0, 0, 0),
        list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
        p=0.3,
    )
    # To torch tensor
    list_images = utils.to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = utils.to_tensor(list_images)
    return list_images


class DosePredictionData(data.Dataset):
    def __init__(self, data_paths, num_samples_per_epoch, phase):
        # 'train' or 'val'
        self.data_paths = data_paths
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {"train": train_transform, "val": val_transform}

        self.list_case_id = self.data_paths[phase]

        random.shuffle(self.list_case_id)
        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            case_id = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            case_id = self.list_case_id[new_index_]

        dict_images = utils.read_image_data(case_id)
        list_images = utils.concatenate(dict_images)

        list_images = self.transform[self.phase](list_images)
        return list_images

    def __len__(self):
        return self.num_samples_per_epoch


def get_loader(
    data_paths,
    train_bs=1,
    val_bs=1,
    train_num_samples_per_epoch=1,
    val_num_samples_per_epoch=1,
    num_workers=0,
):
    train_dataset = DosePredictionData(
        data_paths, num_samples_per_epoch=train_num_samples_per_epoch, phase="train"
    )
    val_dataset = DosePredictionData(
        data_paths, num_samples_per_epoch=val_num_samples_per_epoch, phase="val"
    )

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
