import os
import SimpleITK as sitk
import astra.viz.utils as utils
import astra.data.utils as data_utils


def generate_random_heatmaps(surface_path, save_path):
    target_image = sitk.ReadImage(surface_path)
    target_mask = sitk.GetArrayFromImage(target_image)

    smooth_factor = 0.5
    random_surface = utils.create_random_surface(target_mask, (0, 255), smooth_factor)
    random_image = sitk.GetImageFromArray(random_surface)
    random_image = data_utils.copy_image_info(target_image, random_image)

    sitk.WriteImage(random_image, save_path)
    return


if __name__ == "__main__":
    root_path = "/Users/amithkamath/"
    data_path = os.path.join(root_path, "data/DLDP/astute-oar/DLDP_070/")
    surface_path = os.path.join(data_path, "Target.nii.gz")
    save_path = os.path.join(data_path, "astute-random-target.nii.gz")
    generate_random_heatmaps(surface_path, save_path)
