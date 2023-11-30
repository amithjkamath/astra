import os
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt

from astra.data.utils import read_image_data, concatenate
from monai.networks.nets import BasicUNet

from pathlib import Path
from monai.config import print_config

warnings.filterwarnings("ignore")


def main():
    print_config()

    repo_root = "/Users/amithkamath/repo/astra/"
    data_root = "/Users/amithkamath/data/DLDP/"
    data_path = os.path.join(data_root, "ground_truth_small")
    out_path = os.path.join(repo_root, "monai_unet_results")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    list_test_dirs = [
        os.path.join(data_path, "DLDP_") + str(i).zfill(3) for i in range(81, 91)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(
        spatial_dims=3,
        in_channels=15,
        out_channels=1,
        features=[32, 64, 128, 256, 512, 32],
    ).to(device)

    checkpoint = torch.load(
        os.path.join(
            repo_root, "output/monai_testing/11-29-23_00-38/unet_dose_prediction.pt"
        ),
        map_location=device,
    )
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        step = 1
        for test_dir in list_test_dirs:
            test_id = test_dir.split("/")[-1]
            dict_images = read_image_data(test_dir)
            list_images = concatenate(dict_images)

            input = np.expand_dims(list_images[0], 0)
            gt_dose = np.expand_dims(list_images[1], 0)
            possible_dose_mask = np.expand_dims(list_images[2], 0)

            # forward pass
            input_t = torch.from_numpy(input)
            prediction = model(input_t)

            # save volume slices according to volume name given by fname
            prediction[
                np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)
            ] = 0
            prediction = 70.0 * prediction

            print(step, "   volume out of", len(list_test_dirs), "done.", "\r", end="")
            step += 1

            slice = 16
            # visualize
            fig = plt.figure(figsize=(14, 7))
            plt.title(f"Subject: {test_id}")
            ax = fig.add_subplot(121)
            ax.imshow(gt_dose[0, 0, slice, :, :], "gray")
            ax.set_title("ground truth dose")
            ax.axis("off")

            ax = fig.add_subplot(122)
            ax.imshow(prediction[0, 0, slice, :, :], "gray")
            ax.set_title("predicted dose")
            ax.axis("off")

            plt.show()


if __name__ == "__main__":
    main()
