"""
Reading CT and RTSS data from .dcm files
"""

import glob
import os
from tqdm import tqdm

import pydicom
import pymedphys
import SimpleITK as sitk


def rtdose_to_nifti(base_input_path, base_output_path):
    """
    RTDOSE_TO_NIFTI converts RD*.dcm RT Dose files to NIfTI volumes.
    """
    try:
        fpath = os.path.join(base_input_path)

        rtdose_files = glob.glob(fpath + "//RD*.dcm")
        n_doses = len(rtdose_files)
        for idx in range(n_doses):
            rtdose_file = rtdose_files[idx]
            ds = pydicom.dcmread(rtdose_file)
            dose_image_sitk = sitk.ReadImage(rtdose_file)
            (dose_axes, dose_array) = pymedphys.dicom.zyx_and_dose_from_dataset(ds)
            dose_image = sitk.GetImageFromArray(dose_array)
            dose_image.CopyInformation(dose_image_sitk)

            ct_file = os.path.join(base_output_path, "CT.nii.gz")
            ct_image = sitk.ReadImage(ct_file)

            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator = sitk.sitkLinear
            resample.SetOutputDirection = ct_image.GetDirection()
            resample.SetOutputOrigin(ct_image.GetOrigin())
            resample.SetOutputSpacing(ct_image.GetSpacing())
            resample.SetSize(ct_image.GetSize())

            new_dose_image = resample.Execute(dose_image)
            subject_output_path = os.path.join(base_output_path)
            sitk.WriteImage(
                new_dose_image,
                os.path.join(subject_output_path, "Dose_" + str(idx) + ".nii.gz"),
            )
    except Exception as ex:
        print(ex)
        print("Errored :(")


if __name__ == "__main__":
    input_path = "/Users/amithkamath/repo/deep-planner/data/raw_ONL_Perturbations_084/"
    output_path = (
        "/Users/amithkamath/repo/deep-planner/data/interim_ONL_Perturbations_084/"
    )
    rtdose_to_nifti(input_path, output_path)
