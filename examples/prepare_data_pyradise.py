import os
from tqdm import tqdm
import SimpleITK as sitk

from typing import Tuple, Optional

import pyradise.data as ps_data
import pyradise.fileio as ps_io
import pyradise.process as ps_proc


def get_pipeline(
    output_size: Tuple[int, int, int] = (128, 128, 128),
    output_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    reference_modality: str = "CT",
) -> ps_proc.FilterPipeline:
    # Create an empty filter pipeline
    pipeline = ps_proc.FilterPipeline()

    # Add an orientation filter to the pipeline
    orientation_params = ps_proc.OrientationFilterParams(output_orientation="RAS")
    pipeline.add_filter(ps_proc.OrientationFilter(), orientation_params)

    # Add a resampling filter to the pipeline
    resample_params = ps_proc.ResampleFilterParams(
        output_size,
        output_spacing,
        reference_modality=reference_modality,
        centering_method="reference",
    )
    pipeline.add_filter(ps_proc.ResampleFilter(), resample_params)

    return pipeline


def convert_dicom_to_nifti_with_modality_config(
    input_path: str, output_path: str
) -> None:
    # Instantiate a new loader
    loader = ps_io.SubjectLoader()

    # (optional)
    # Get the filter pipeline
    pipeline = get_pipeline()

    # Instantiate a new writer with default settings
    # Note: You can adjust here the output image file format
    # and the naming of the output files
    writer = ps_io.SubjectWriter()

    # (optional)
    # Instantiate a new selection to exclude additional SeriesInfo entries
    expected_modalities = ("FLAIR", "CT", "T1", "T2", "T1c")
    modality_selection = ps_io.ModalityInfoSelector(expected_modalities)

    # Search DICOM files for each subject and iterate over the crawler
    crawler = ps_io.DatasetDicomCrawler(input_path)
    for series_info in crawler:
        # (optional)
        # Keep just the selected modalities for loading
        # Note: SeriesInfo entries for non-image data get returned unfiltered
        series_info = modality_selection.execute(series_info)

        # Load the subject from the series info
        subject = loader.load(series_info)

        # (optional)
        # Execute the filter pipeline on the subject
        print(f"Processing subject {subject.get_name()}...")
        subject = pipeline.execute(subject)

        # Write each subject to a specific subject directory
        writer.write_to_subject_folder(output_path, subject, write_transforms=False)


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

        ## Run this first to generate the modality configuration
        # ps_io.DatasetDicomCrawler(input_folder, write_modality_config=True).execute()

        # Execute the conversion procedure (approach 1)
        convert_dicom_to_nifti_with_modality_config(input_folder, output_folder)
