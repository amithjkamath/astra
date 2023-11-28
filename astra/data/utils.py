import os
import numpy as np
import SimpleITK as sitk


def read_image_data(patient_dir: str):
    dict_images = {}
    list_structures = [
        "CT",
        "Brain",
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
        "Target",
        "Dose",
        "Dose_Mask",
    ]

    for structure_name in list_structures:
        structure_file = patient_dir + "/" + structure_name + ".nii.gz"

        if structure_name == "CT":
            dtype = sitk.sitkInt16
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            dict_images[structure_name] = sitk.GetArrayFromImage(
                dict_images[structure_name]
            )[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images


def concatenate(dict_images: dict):
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

    # Possible mask
    possible_dose_mask = dict_images["Brain"]

    list_images = [
        np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
        possible_dose_mask,
    ]
    return list_images


def copy_image_info(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


# Input is C*Z*H*W
def flip_3d(input, list_axes):
    if "Z" in list_axes:
        input = input[:, ::-1, :, :]
    if "W" in list_axes:
        input = input[:, :, :, ::-1]

    return input


def resample(input_image: sitk.Image, reference_image: sitk.Image):
    """
    RESAMPLE resamples input image to be in the same coordinates as reference_image.
    """

    rs_filter = sitk.ResampleImageFilter()
    rs_filter.SetInterpolator = sitk.sitkLinear
    rs_filter.SetOutputDirection = reference_image.GetDirection()
    rs_filter.SetOutputOrigin(reference_image.GetOrigin())
    rs_filter.SetOutputSpacing(reference_image.GetSpacing())
    rs_filter.SetSize(reference_image.GetSize())

    resampled_image = rs_filter.Execute(input_image)
    return resampled_image
