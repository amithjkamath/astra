"""
Reading CT and RTSS data from .dcm files.
See https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/Examples/DICOMRTTool_Tutorial.ipynb for more.
"""

import os

from DicomRTTool.ReaderWriter import DicomReaderWriter
import SimpleITK as sitk


def ct_to_nifti(base_input_path, base_output_path, out_fname):
    """
    CT_TO_NIFTI converts CT DICOM volumes to NIfTI.
    """

    fpath = os.path.join(base_input_path)
    dicom_reader = DicomReaderWriter(description="Examples", arg_max=True)
    dicom_reader.walk_through_folders(fpath)
    # all_rois = dicom_reader.return_rois(print_rois=True)

    dicom_reader.set_contour_names_and_associations(
        Contour_Names=["Brain"], associations={"brain": "Brain"}
    )
    indexes = (
        dicom_reader.which_indexes_have_all_rois()
    )  # Check to see which indexes have all of the rois we want, now we can see indexes
    pt_indx = indexes[-1]
    dicom_reader.set_index(
        pt_indx
    )  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    dicom_reader.get_images_and_mask()  # Load up the images and mask for the requested index

    image = dicom_reader.ArrayDicom  # image array
    mask = dicom_reader.mask  # mask array

    out_path = os.path.join(base_output_path)
    os.makedirs(out_path, exist_ok=True)
    image_sitk_handle = dicom_reader.dicom_handle
    sitk.WriteImage(image_sitk_handle, os.path.join(out_path, out_fname))
    return


def rtss_to_nifti(
    base_input_path,
    base_output_path,
    out_fname,
    contour_names,
    associations,
):
    """
    RTSS_TO_NIFTI converts RTSS DICOM volumes to NIfTI.
    """
    fpath = os.path.join(base_input_path)
    # rtstruct_file = glob.glob(fpath + "//RS*.dcm")[0]
    dicom_reader = DicomReaderWriter(description="Examples", arg_max=True)
    dicom_reader.walk_through_folders(fpath)
    # all_rois = dicom_reader.return_rois(print_rois=True)

    dicom_reader.set_contour_names_and_associations(
        Contour_Names=contour_names, associations=associations
    )
    indexes = (
        dicom_reader.which_indexes_have_all_rois()
    )  # Check to see which indexes have all of the rois we want, now we can see indexes
    pt_indx = indexes[-1]
    dicom_reader.set_index(
        pt_indx
    )  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    dicom_reader.get_images_and_mask()  # Load up the images and mask for the requested index

    image = dicom_reader.ArrayDicom  # image array
    mask = dicom_reader.mask  # mask array

    out_path = os.path.join(base_output_path)
    os.makedirs(out_path, exist_ok=True)
    mask_sitk_handle = dicom_reader.annotation_handle  # SimpleITK mask handle
    sitk.WriteImage(mask_sitk_handle, os.path.join(out_path, out_fname))
    return


def rt_to_nifti(input_folder, output_folder):
    """
    RT_TO_NIFTI converts RT data in base_input_folder to NIfTI in base_output_folder.
    """
    try:
        ct_to_nifti(input_folder, output_folder, "CT.nii.gz")

        fpath = os.path.join(input_folder)
        dicom_reader = DicomReaderWriter(description="Examples", arg_max=True)
        dicom_reader.walk_through_folders(fpath)
        all_rois = dicom_reader.return_rois(print_rois=True)

        for roi in all_rois:
            contour_names = [roi]
            associations = {roi: roi}
            rtss_to_nifti(
                input_folder,
                output_folder,
                roi + ".nii.gz",
                contour_names,
                associations,
            )

    except Exception as ex:
        print(ex)
        print("Errored.")


if __name__ == "__main__":
    input_path = "/Users/amithkamath/repo/deep-planner/data/raw_ONL_Perturbations_084/"
    output_path = (
        "/Users/amithkamath/repo/deep-planner/data/interim_ONL_Perturbations_084/"
    )
    rt_to_nifti(input_path, output_path)
