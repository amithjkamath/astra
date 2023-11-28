import glob

import pydicom
import pymedphys
import SimpleITK as sitk
from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass


def ct_to_nifti(input_path):
    """
    CT_TO_NIFTI converts CT DICOM volumes to NIfTI.
    """

    dicom_reader = DicomReaderWriter(description="Examples", arg_max=True)
    dicom_reader.walk_through_folders(input_path)

    associations = [ROIAssociationClass("brain", ["Brain"])]
    dicom_reader.set_contour_names_and_associations(
        contour_names=["Brain"], associations=associations
    )
    indexes = (
        dicom_reader.which_indexes_have_all_rois()
    )  # Check to see which indexes have all of the rois we want, now we can see indexes
    pt_indx = indexes[-1]
    dicom_reader.set_index(
        pt_indx
    )  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    dicom_reader.get_images_and_mask()  # Load up the images and mask for the requested index
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


def resample(input_image: sitk.Image, reference_image: sitk.Image):

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = sitk.sitkLinear
    resample.SetOutputDirection = reference_image.GetDirection()
    resample.SetOutputOrigin(reference_image.GetOrigin())
    resample.SetOutputSpacing(reference_image.GetSpacing())
    resample.SetSize(reference_image.GetSize())

    resampled_image = resample.Execute(input_image)
    return resampled_image
