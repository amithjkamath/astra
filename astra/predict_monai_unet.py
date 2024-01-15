import torch
import warnings
import matplotlib.pyplot as plt

from monai.networks.nets import BasicUNet
from monai.config import print_config

from astra.data.utils import read_image_data, concatenate, copy_image_info
from astra.train.evaluate_DLDP import *

warnings.filterwarnings("ignore")


def main():
    print_config()

    repo_root = "/Users/amithkamath/repo/astra/"
    data_root = "/Users/amithkamath/data/DLDP/"
    data_path = os.path.join(data_root, "astute-results")

    patient_id = "0800"
    list_test_dirs = [os.path.join(data_path, "DLDP_" + patient_id)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(
        spatial_dims=3,
        in_channels=15,
        out_channels=1,
        features=[32, 64, 128, 256, 512, 32],
    ).to(device)

    checkpoint = torch.load(
        os.path.join(repo_root, "models/unet-monai/unet_dose_prediction.pt"),
        map_location=device,
    )
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        for test_dir in list_test_dirs:
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

            # Save prediction to nii image
            template_nii = sitk.ReadImage(test_dir + "/Dose_Mask.nii.gz")
            prediction_nii = sitk.GetImageFromArray(prediction[0, 0, ...])
            prediction_nii = copy_image_info(template_nii, prediction_nii)
            if not os.path.exists(test_dir + "/Prediction"):
                os.mkdir(test_dir + "/Prediction")
            sitk.WriteImage(
                prediction_nii, test_dir + "/Prediction/Dose.nii.gz",
            )

            Dose_score, DVH_score = get_Dose_score_and_DVH_score_per_ROI(
                prediction_dir=test_dir + "/Prediction",
                patient_id=patient_id,
                gt_dir=data_path,
            )

            print("Dose score is: " + str(Dose_score))
            print("DVH score is: " + str(DVH_score))

            plt.show()


if __name__ == "__main__":
    main()
