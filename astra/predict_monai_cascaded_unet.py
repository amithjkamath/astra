import os
import numpy as np
import torch
import warnings
import SimpleITK as sitk

from astra.data.utils import read_image_data, concatenate, copy_image_info
from astra.model.CascadedUNet import CascadedUNet

from pathlib import Path
from monai.config import print_config

warnings.filterwarnings("ignore")


def main():
    print_config()

    repo_root = "/home/akamath/Documents/astra/"
    data_root = "/home/akamath/Documents/astra/data/processed-to-train/"
    data_path = os.path.join(data_root)
    out_path = os.path.join(repo_root, "monai_unet_results")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    list_test_dirs = [
        os.path.join(data_path, "ISAS_GBM_") + str(i).zfill(3) for i in range(81, 91)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CascadedUNet(
        spatial_dims=3,
        in_channels=15,
        out_channels=1,
        channels_first=(16, 32, 64, 128, 256),
        channels_second=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        act="ReLU",
    ).to(device)

    checkpoint = torch.load(
        os.path.join(
            repo_root,
            "output/monai_testing/01-12-24_17-50/cascaded_unet_dose_prediction.pt",
        ),
        map_location=device,
    )
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        for test_dir in list_test_dirs:
            test_id = test_dir.split("/")[-1]
            dict_images = read_image_data(test_dir)
            list_images = concatenate(dict_images)

            input = np.expand_dims(list_images[0], 0)
            gt_dose = np.expand_dims(list_images[1], 0)
            possible_dose_mask = np.expand_dims(list_images[2], 0)

            # forward pass
            input_t = torch.from_numpy(input)
            prediction = model(input_t.to(device))

            prediction = prediction.cpu()

            # save volume slices according to volume name given by fname
            prediction[
                np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)
            ] = 0
            prediction = 70.0 * prediction

            # Save prediction to nii image
            template_nii = sitk.ReadImage(test_dir + "/Dose.nii.gz")
            prediction_nii = sitk.GetImageFromArray(prediction[0, 0, :, :, :])
            prediction_nii = copy_image_info(template_nii, prediction_nii)
            if not os.path.exists(test_dir + "/" + test_id):
                os.mkdir(test_dir + "/" + test_id)
            sitk.WriteImage(
                prediction_nii, test_dir + "/" + test_id + "/Predicted_Dose.nii.gz"
            )


if __name__ == "__main__":
    main()
