# -*- encoding: utf-8 -*-
import sys
import argparse

from astra.data.utils import read_image_data, concatenate, copy_image_info

from astra.model.C3D import Model, inference
from astra.train.network_trainer import *
from astra.train.evaluate_DLDP import *

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))


def run(trainer, list_patient_dirs, save_path, do_TTA=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs):
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
            prediction = 70.0 * prediction

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask.nii.gz")
            prediction_nii = sitk.GetImageFromArray(prediction)
            prediction_nii = copy_image_info(templete_nii, prediction_nii)
            if not os.path.exists(save_path + "/" + patient_id):
                os.mkdir(save_path + "/" + patient_id)
            sitk.WriteImage(
                prediction_nii, save_path + "/" + patient_id + "/Dose.nii.gz"
            )


if __name__ == "__main__":

    root_dir = "/Users/amithkamath/repo/astra"
    model_dir = os.path.join(root_dir, "models/isometric")

    gt_dir = os.path.join("/Users/amithkamath/data/DLDP/astute-results")
    test_dir = gt_dir  # change this if somewhere else.

    for patient_id in ["0800"]:

        output_dir = os.path.join(gt_dir, "DLDP_" + patient_id)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(model_dir):
            raise Exception(
                "OpenKBP_C3D should be prepared before testing, please run prepare_OpenKBP_C3D.py"
            )

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--GPU_id",
            type=int,
            default=-1,
            help="GPU id used for testing (default: 0)",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=os.path.join(model_dir, "best_train_loss.pkl"),
        )
        parser.add_argument(
            "--TTA",
            type=bool,
            default=True,
            help="do test-time augmentation, default True",
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

        # Start inference
        print("\n\n# Start inference !")
        list_patient_dirs = [os.path.join(gt_dir, "DLDP_" + patient_id)]
        run(
            trainer,
            list_patient_dirs,
            save_path=os.path.join(trainer.setting.output_dir, "Prediction"),
            do_TTA=args.TTA,
        )

        # Evaluation
        print("\n\n# Start evaluation !")
        Dose_score, DVH_score = get_Dose_score_and_DVH_score_per_ROI(
            prediction_dir=os.path.join(trainer.setting.output_dir, "Prediction"),
            patient_id=patient_id,
            gt_dir=gt_dir,
        )

        print("Dose score is: " + str(Dose_score))
        print("DVH score is: " + str(DVH_score))
