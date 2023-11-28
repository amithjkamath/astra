# -*- encoding: utf-8 -*-
import os
import sys
import numpy as np
import SimpleITK as sitk
import skimage.morphology as skm
from tqdm import tqdm

if os.path.abspath("../astra") not in sys.path:
    sys.path.insert(0, os.path.abspath("../astra"))

from astra.data.utils import (
    read_image_data,
    copy_sitk_imageinfo,
)


def find_boundary_points(volume):
    """
    Find points on the boundary of a region of interest.
    These points will then be used to create perturbations.
    """
    ball = skm.ball(1)
    volume_larger = skm.binary_erosion(volume[0, :, :, :], ball)
    boundary_volume = volume_larger - volume[0, :, :, :]
    points = np.nonzero(boundary_volume)
    out_points = []

    # Choose 10 here to sub-sample the surface. Need to think of a better way to do this.
    for idx in range(0, len(points[0]), 100):
        x = points[0][idx]
        y = points[1][idx]
        z = points[2][idx]
        out_points.append([x, y, z])
    return out_points


def oversegment_at(volume, point):
    """
    Dilate the binary volume 'volume' at the point specified by point.
    """
    ball = skm.ball(3)
    point_vol = np.zeros(volume[0, :, :, :].shape, dtype=np.uint8)
    point_vol[point[0], point[1], point[2]] = 1
    volume_out = skm.binary_dilation(point_vol, ball).astype(np.uint8)
    volume_out += volume[0, :, :, :].astype(np.uint8)
    volume_out[volume_out >= 1] = 1
    volume_out = volume_out[np.newaxis, :, :, :]
    return volume_out


def generate_perturbation_masks(list_patient_dirs, save_path):
    """
    This function helps create perturbations in the OAR, and then evaluates the dose.
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for patient_dir in list_patient_dirs:
        patient_id = patient_dir.split("/")[-1]

        dict_images = read_image_data(patient_dir)

        list_OAR_names = ["Target"]

        for oar in list_OAR_names:

            print("Working on: ", oar.split("_")[0])

            template_nii = sitk.ReadImage(patient_dir + "/Brain.nii.gz")
            original_mask = sitk.GetImageFromArray(np.squeeze(dict_images[oar]))
            original_mask = copy_sitk_imageinfo(template_nii, original_mask)
            if not os.path.exists(save_path + "/" + patient_id):
                os.mkdir(save_path + "/" + patient_id)
            sitk.WriteImage(
                original_mask,
                save_path + "/" + patient_id + "/Original_" + oar + ".nii.gz",
            )

            point_set = find_boundary_points(dict_images[oar])

            print("Points on surface: ", len(point_set))

            # At this stage, do perturbation on the OAR boundary.
            for point in tqdm(point_set):

                perturbed_images = read_image_data(patient_dir)
                perturbed_images[oar] = oversegment_at(perturbed_images[oar], point)

                template_nii = sitk.ReadImage(patient_dir + "/Brain.nii.gz")
                perturbed_mask = sitk.GetImageFromArray(
                    np.squeeze(perturbed_images[oar])
                )
                perturbed_mask = copy_sitk_imageinfo(template_nii, perturbed_mask)
                if not os.path.exists(save_path + "/" + patient_id):
                    os.mkdir(save_path + "/" + patient_id)
                sitk.WriteImage(
                    perturbed_mask,
                    os.path.join(
                        save_path,
                        patient_id
                        + "Perturbed_"
                        + oar
                        + "_"
                        + str(point[0])
                        + "_"
                        + str(point[1])
                        + "_"
                        + str(point[2])
                        + ".nii.gz",
                    ),
                )


if __name__ == "__main__":

    root_dir = "/"
    model_dir = os.path.join(root_dir, "models")
    output_dir = os.path.join(root_dir, "output_perturb")
    os.makedirs(output_dir, exist_ok=True)

    gt_dir = os.path.join("/Users/amithkamath/data/DLDP/ground_truth")
    test_dir = gt_dir  # change this if somewhere else.

    for subject_id in [81]:

        list_patient_dirs = [os.path.join(test_dir, "DLDP_" + str(subject_id).zfill(3))]
        generate_perturbation_masks(
            list_patient_dirs, save_path=os.path.join(output_dir, "Prediction"),
        )
