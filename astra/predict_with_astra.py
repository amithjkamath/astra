# -*- encoding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
import skimage.morphology as skm
from tqdm import tqdm

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))

from astra.data.utils import (
    read_image_data,
    concatenate,
    copy_image_info,
)
from astra.model.C3D import Model, inference
from astra.train.network_trainer import *


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
    for idx in range(0, len(points[0]), 5):
        x = points[0][idx]
        y = points[1][idx]
        z = points[2][idx]
        out_points.append([x, y, z])
    return out_points


def dilate_at(volume, point):
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


def inference_with_perturbation(trainer, list_patient_dirs, save_path, do_TTA=True):
    """
    This function helps create perturbations in the OAR, and then evaluates the dose.
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in list_patient_dirs:
            patient_id = patient_dir.split("/")[-1]

            dict_images = read_image_data(patient_dir)

            list_images = concatenate(dict_images)

            input_ = list_images[0]
            possible_dose_mask = list_images[1]

            # Test-time augmentation
            if do_TTA:
                TTA_mode = [[], ["Z"], ["W"], ["Z", "W"]]
            else:
                TTA_mode = [[]]
            prediction = inference(trainer, input_, TTA_mode)

            # Pose-processing
            prediction[
                np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)
            ] = 0
            gt_prediction = 70.0 * prediction

            templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask.nii.gz")
            prediction_nii = sitk.GetImageFromArray(gt_prediction)
            prediction_nii = copy_image_info(templete_nii, prediction_nii)
            if not os.path.exists(save_path + "/" + patient_id):
                os.mkdir(save_path + "/" + patient_id)
            sitk.WriteImage(
                prediction_nii, save_path + "/" + patient_id + "/Dose_gt.nii.gz",
            )

            list_OAR_names = ["Target"]

            for oar in list_OAR_names:

                print("Working on: ", oar.split("_")[0])

                perturb_prediction = np.zeros_like(gt_prediction)

                point_set = find_boundary_points(dict_images[oar])

                print("Points on surface: ", len(point_set))

                # At this stage, do perturbation on the OAR boundary.
                for point in tqdm(point_set):

                    dict_images = read_image_data(patient_dir)
                    dict_images[oar] = dilate_at(dict_images[oar], point)

                    list_images = concatenate(dict_images)

                    input_ = list_images[0]
                    possible_dose_mask = list_images[1]

                    # Test-time augmentation
                    if do_TTA:
                        TTA_mode = [[], ["Z"], ["W"], ["Z", "W"]]
                    else:
                        TTA_mode = [[]]
                    prediction = inference(trainer, input_, TTA_mode)

                    # Pose-processing
                    prediction[
                        np.logical_or(
                            possible_dose_mask[0, :, :, :] < 1, prediction < 0
                        )
                    ] = 0
                    prediction = 70.0 * prediction

                    absdiff = np.sum(np.abs(gt_prediction - prediction))

                    perturb_prediction[point[0], point[1], point[2]] = absdiff

                templete_nii = sitk.ReadImage(patient_dir + "/Brain.nii.gz")
                prediction_nii = sitk.GetImageFromArray(perturb_prediction)
                prediction_nii = copy_image_info(templete_nii, prediction_nii)
                if not os.path.exists(save_path + "/" + patient_id):
                    os.mkdir(save_path + "/" + patient_id)
                sitk.WriteImage(
                    prediction_nii,
                    save_path + "/" + patient_id + "/Perturbed_" + oar + ".nii.gz",
                )


if __name__ == "__main__":

    root_dir = "/Users/amithkamath/repo/astra"
    model_dir = os.path.join(root_dir, "models")
    output_dir = os.path.join(root_dir, "output_perturb")
    os.makedirs(output_dir, exist_ok=True)

    gt_dir = os.path.join("/Users/amithkamath/data/DLDP/ground_truth")
    test_dir = gt_dir  # change this if somewhere else.

    if not os.path.exists(model_dir):
        raise Exception(
            "OpenKBP_C3D should be prepared before testing, please run prepare_OpenKBP_C3D.py"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU_id", type=int, default=-1, help="GPU id used for testing (default: 0)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(model_dir, "best_val_evaluation_index.pkl"),
    )
    parser.add_argument(
        "--TTA", type=bool, default=True, help="do test-time augmentation, default True"
    )
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = "C3D"
    trainer.setting.output_dir = output_dir

    trainer.setting.network = Model(
        in_ch=15,
        out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        list_ch_B=[-1, 32, 64, 128, 256, 512],
    )

    # Load model weights
    trainer.init_trainer(
        ckpt_file=args.model_path, list_GPU_ids=[args.GPU_id], only_network=True
    )

    for subject_id in [81]:

        # Start inference
        print("\n\n# Start inference !")
        list_patient_dirs = [os.path.join(test_dir, "DLDP_" + str(subject_id).zfill(3))]
        inference_with_perturbation(
            trainer,
            list_patient_dirs,
            save_path=os.path.join(trainer.setting.output_dir, "Prediction"),
            do_TTA=args.TTA,
        )
