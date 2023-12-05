import os

import SimpleITK as sitk
import numpy as np
import pyvista as pv

from astra.data.utils import read_image_data


def main():

    data_root = "/Users/amithkamath/data/DLDP/perturb-dldp"
    gt_path = os.path.join(data_root, "ground_truth")
    astra_path = os.path.join(data_root, "astra")

    list_test_dirs = [
        os.path.join(astra_path, "DLDP_") + str(i).zfill(3) for i in range(81, 82)
    ]
    for test_dir in list_test_dirs:
        test_id = test_dir.split("/")[-1]

        p = pv.Plotter()
        p.clear()

        mesh = pv.Cube(
            center=(64, 64, 64), x_length=128.0, y_length=128.0, z_length=128.0,
        )
        p.add_mesh(mesh, color="red", style="wireframe", line_width=2)

        target_file = os.path.join(gt_path, test_id, "Target.nii.gz")
        target_image = sitk.ReadImage(target_file, sitk.sitkUInt8)
        target_array = sitk.GetArrayFromImage(target_image)
        p.add_volume(np.multiply(target_array, 255), cmap="jet", shade=False)

        scaled_volume = {}
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
            structure_file = os.path.join(test_dir, "astra_" + structure + ".nii.gz")
            struct_image = sitk.ReadImage(structure_file, sitk.sitkFloat32)
            struct_array = sitk.GetArrayFromImage(struct_image)
            max_value = np.max(struct_array)
            scaled_volume[structure] = struct_array * 255.0 / max_value
            p.add_volume(
                scaled_volume[structure].astype(np.uint8), shade=False, cmap="viridis",
            )

        p.show_grid()
        p.show()
        p.deep_clean()


if __name__ == "__main__":
    main()
