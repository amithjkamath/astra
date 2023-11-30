import numpy as np
from skimage.morphology import dilation, binary_erosion, ball
from skimage.filters import gaussian


def create_astra_surface(
    discrete_perturb, structure_volume, expand_factor=5, smooth_factor=5
):

    # First dilate the voxels in the image by expand_factor
    expand_strel = ball(expand_factor)
    dilated_perturb = dilation(discrete_perturb, expand_strel)

    # then smooth the values in the volume by smooth_factor
    smooth_perturb = gaussian(dilated_perturb, sigma=smooth_factor)

    # then store values only on the contour surface of structure_volume
    contour_strel = ball(3)
    eroded_structure = binary_erosion(structure_volume, contour_strel)
    contour_region = structure_volume - eroded_structure

    astra_volume = np.zeros(shape=discrete_perturb.shape, dtype=np.float32)
    astra_volume[contour_region == True] = smooth_perturb[contour_region == True]

    return astra_volume
