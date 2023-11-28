import os
from tqdm import tqdm
import SimpleITK as sitk

import astra.utils.convert as convert

if __name__ == "__main__":
    input_path = "/Users/amithkamath/data/DLDP/raw/"
    output_path = "/Users/amithkamath/data/DLDP/processed-dose"
    num_subjects = 5

    for subject_id in tqdm(range(1, num_subjects + 1)):
        str_id = str(subject_id).zfill(3)
        subject_name = "DLDP_" + str_id

        input_folder = os.path.join(input_path, subject_name)
        output_folder = os.path.join(output_path, subject_name)
        os.makedirs(output_folder, exist_ok=True)

        ct_image = convert.ct_to_nifti(input_folder)
        sitk.WriteImage(ct_image, os.path.join(output_folder, "CT.nii.gz"))

        dose_image = convert.rtdose_to_nifti(input_folder)
        sitk.WriteImage(dose_image, os.path.join(output_folder, "Dose.nii.gz"))
