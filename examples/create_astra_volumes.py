import os
import SimpleITK as sitk
import astra.viz.utils as vizutils
import astra.data.utils as datautils


def main():

    data_root = "/Users/amithkamath/data/DLDP/perturb-dldp"
    input_path = os.path.join(data_root, "Prediction")
    gt_path = os.path.join(data_root, "ground_truth")
    output_path = os.path.join(data_root, "astra")
    os.makedirs(output_path, exist_ok=True)

    cases = os.listdir(input_path)
    for case in cases:
        if os.path.isdir(os.path.join(input_path, case)):
            test_id = case.split("/")[-1]
            os.makedirs(os.path.join(output_path, case), exist_ok=True)
            for structure in [
                "BrainStem",
                "Chiasm",
                "Cochlea_L",
                "Cochlea_R",
                "Eye_L",
                "Eye_R",
                "Hippocampus_L",
                "Hippocampus_R",
                "LacrimalGland_L",
                "LacrimalGland_R",
                "OpticNerve_L",
                "OpticNerve_R",
                "Pituitary",
            ]:
                perturbed_volume = sitk.ReadImage(
                    os.path.join(
                        input_path, case, "Perturbed_" + structure + ".nii.gz"
                    ),
                    sitk.sitkFloat32,
                )
                perturbed_array = sitk.GetArrayFromImage(perturbed_volume)

                structure_volume = sitk.ReadImage(
                    os.path.join(gt_path, case, structure + ".nii.gz"),
                    sitk.sitkFloat32,
                )
                structure_array = sitk.GetArrayFromImage(structure_volume)

                astra_array = vizutils.create_astra_surface(
                    perturbed_array, structure_array, expand_factor=5, smooth_factor=5
                )
                astra_volume = sitk.GetImageFromArray(astra_array)

                astra_volume = datautils.copy_image_info(structure_volume, astra_volume)
                sitk.WriteImage(
                    astra_volume,
                    os.path.join(output_path, case, "astra_" + structure + ".nii.gz"),
                )


if __name__ == "__main__":
    main()
