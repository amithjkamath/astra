import glob

import pydicom
import pymedphys
import SimpleITK as sitk
from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass


def images_to_nifti(input_path):
    """
    IMAGES_TO_NIFTI converts DICOM image volumes to NIfTI.
    """

    dicom_reader = DicomReaderWriter(arg_max=True)
    dicom_reader.walk_through_folders(input_path)

    images = []
    indices = dicom_reader.indexes_with_contours
    for index in indices:
        dicom_reader.set_index(index)
        dicom_reader.get_images()
        image_handle = dicom_reader.dicom_handle
        images.append(image_handle)
    return images


def ct_to_nifti(input_path):
    """
    CT_TO_NIFTI converts CT DICOM volumes to NIfTI.
    This works when the only image data in the folder is CT.
    """

    dicom_reader = DicomReaderWriter(arg_max=True)
    dicom_reader.walk_through_folders(input_path)

    associations = [ROIAssociationClass("brain", ["Brain"])]
    dicom_reader.set_contour_names_and_associations(
        contour_names=["Brain"], associations=associations
    )
    indexes = dicom_reader.which_indexes_have_all_rois()
    pt_indx = indexes[-1]
    dicom_reader.set_index(pt_indx)
    dicom_reader.get_images_and_mask()
    image_handle = dicom_reader.dicom_handle
    return image_handle


def rtdose_to_nifti(input_path):
    """
    RTDOSE_TO_NIFTI converts RD*.dcm RT Dose files to NIfTI volumes.
    """

    rtdose_file = glob.glob(input_path + "//RD*.dcm")

    ds = pydicom.dcmread(rtdose_file[0])
    dose_image_sitk = sitk.ReadImage(rtdose_file[0])
    (dose_axes, dose_array) = pymedphys.dicom.zyx_and_dose_from_dataset(ds)
    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.CopyInformation(dose_image_sitk)
    return dose_image
