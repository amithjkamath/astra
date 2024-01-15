import os
from tqdm import tqdm
import SimpleITK as sitk

import astra.utils.convert as convert

if __name__ == "__main__":
    input_path = "/Users/amithkamath/data/DLDP/ISAS_GBM_data/"
    output_path = "/Users/amithkamath/data/DLDP/ISAS_GBM_processed"
    num_subjects = 5

    for subject_id in tqdm([81, 83, 85, 87, 89]):
        str_id = str(subject_id).zfill(3)
        subject_name = "ISAS_GBM_" + str_id

        input_folder = os.path.join(input_path, subject_name)
        output_folder = os.path.join(output_path, subject_name)
        os.makedirs(output_folder, exist_ok=True)

        image_data = convert.images_to_nifti(input_folder)
        seq = 1
        for image in image_data:
            sitk.WriteImage(
                image, os.path.join(output_folder, "image_" + str(seq) + ".nii.gz")
            )
            seq += 1

        """
        ct_image = convert.ct_to_nifti(input_folder)
        sitk.WriteImage(ct_image, os.path.join(output_folder, "CT.nii.gz"))

        ct_image = convert.mr_to_nifti(input_folder, "t1")
        sitk.WriteImage(ct_image, os.path.join(output_folder, "T1.nii.gz"))
        """

        # dose_image = convert.rtdose_to_nifti(input_folder)
        # sitk.WriteImage(dose_image, os.path.join(output_folder, "Dose.nii.gz"))
