# -*- encoding: utf-8 -*-
import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2
import torch


# Random flip
def random_flip_3d(list_images, list_axis=(0, 1, 2), p=0.5):
    if random.random() <= p:
        if 0 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, ::-1, :, :]
        if 1 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, ::-1, :]
        if 2 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, :, ::-1]

    return list_images


# Random rotation using OpenCV
def random_rotate_around_z_axis(
    list_images, list_angles, list_interp, list_boder_value, p=0.5
):
    if random.random() <= p:
        # Randomly pick an angle list_angles
        _angle = random.sample(list_angles, 1)[0]
        # Do not use random scaling, set scale factor to 1
        _scale = 1.0

        for image_i in range(len(list_images)):
            for chan_i in range(list_images[image_i].shape[0]):
                for slice_i in range(list_images[image_i].shape[1]):
                    rows, cols = list_images[image_i][chan_i, slice_i, :, :].shape
                    M = cv2.getRotationMatrix2D(
                        ((cols - 1) / 2.0, (rows - 1) / 2.0), _angle, scale=_scale
                    )
                    list_images[image_i][chan_i, slice_i, :, :] = cv2.warpAffine(
                        list_images[image_i][chan_i, slice_i, :, :],
                        M,
                        (cols, rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=list_boder_value[image_i],
                        flags=list_interp[image_i],
                    )
    return list_images


# Random translation
def random_translate(list_images, roi_mask, p, max_shift, list_pad_value):
    if random.random() <= p:
        exist_mask = np.where(roi_mask > 0)
        ori_z, ori_h, ori_w = list_images[0].shape[1:]

        bz = min(max_shift - 1, np.min(exist_mask[0]))
        ez = max(ori_z - 1 - max_shift, np.max(exist_mask[0]))
        bh = min(max_shift - 1, np.min(exist_mask[1]))
        eh = max(ori_h - 1 - max_shift, np.max(exist_mask[1]))
        bw = min(max_shift - 1, np.min(exist_mask[2]))
        ew = max(ori_w - 1 - max_shift, np.max(exist_mask[2]))

        for image_i in range(len(list_images)):
            list_images[image_i] = list_images[image_i][
                :, bz : ez + 1, bh : eh + 1, bw : ew + 1
            ]

        # Pad to original size
        list_images = random_pad_to_size_3d(
            list_images,
            target_size=[ori_z, ori_h, ori_w],
            list_pad_value=list_pad_value,
        )
    return list_images


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Pad
def random_pad_to_size_3d(list_images, target_size, list_pad_value):
    _, ori_z, ori_h, ori_w = list_images[0].shape[:]
    new_z, new_h, new_w = target_size[:]

    pad_z = new_z - ori_z
    pad_h = new_h - ori_h
    pad_w = new_w - ori_w

    pad_z_1 = random.randint(0, pad_z)
    pad_h_1 = random.randint(0, pad_h)
    pad_w_1 = random.randint(0, pad_w)

    pad_z_2 = pad_z - pad_z_1
    pad_h_2 = pad_h - pad_h_1
    pad_w_2 = pad_w - pad_w_1

    output = []
    for image_i in range(len(list_images)):
        _image = list_images[image_i]
        output.append(
            np.pad(
                _image,
                ((0, 0), (pad_z_1, pad_z_2), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                mode="constant",
                constant_values=list_pad_value[image_i],
            )
        )
    return output


"""
images are always C*Z*H*W
"""


def read_data(patient_dir):
    dict_images = {}
    list_structures = [
        "CT",
        "Dose_Mask",
        "BrainStem",
        "Chiasm",
        "Cochlea_L",
        "Cochlea_R",
        "Eye_L",
        "Eye_R",
        "Hippocampus_L",
        "Hippocampus_R",
        "LacrimalGland_L",
        "LacrimalGland_R",
        "OpticNerve_L",
        "OpticNerve_R",
        "Pituitary",
        "Dose",
        "Target",
    ]

    for structure_name in list_structures:
        structure_file = patient_dir + "/" + structure_name + ".nii.gz"

        if structure_name == "CT":
            dtype = sitk.sitkInt16
        elif structure_name == "Dose":
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            # To numpy array (C * Z * H * W)
            dict_images[structure_name] = sitk.GetArrayFromImage(
                dict_images[structure_name]
            )[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images


def pre_processing(dict_images):
    # PTVs
    PTVs = dict_images["Target"]

    # OARs
    list_OAR_names = [
        "BrainStem",
        "Chiasm",
        "Cochlea_L",
        "Cochlea_R",
        "Eye_L",
        "Eye_R",
        "Hippocampus_L",
        "Hippocampus_R",
        "LacrimalGland_L",
        "LacrimalGland_R",
        "OpticNerve_L",
        "OpticNerve_R",
        "Pituitary",
    ]
    OAR_all = np.concatenate(
        [dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0
    )

    # CT image
    CT = dict_images["CT"]
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.0

    # Dose image
    dose = dict_images["Dose"] / 70.0

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images["Dose_Mask"]

    list_images = [
        np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
        dose,  # Label
        possible_dose_mask,
    ]
    return list_images


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = random_rotate_around_z_axis(
        list_images,
        list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
        list_boder_value=(0, 0, 0),
        list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
        p=0.3,
    )
    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
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

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images)

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
    num_works=0,
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
        num_workers=num_works,
        pin_memory=False,
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_works,
        pin_memory=False,
    )

    return train_loader, val_loader
