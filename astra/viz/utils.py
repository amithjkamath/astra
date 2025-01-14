import numpy as np
from skimage.morphology import dilation, erosion, binary_erosion, ball
from skimage.filters import gaussian
from skimage.util import random_noise
from scipy.ndimage.filters import gaussian_filter


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


def create_random_surface(surface_mask, value_limits, smooth_factor):
    input_size = surface_mask.shape

    if len(input_size) == 3:
        random_surface = np.random.rand(input_size[0], input_size[1], input_size[2])
    else:
        random_surface = np.random.rand(input_size[0], input_size[1])

    random_surface = gaussian_filter(random_surface, smooth_factor)
    random_surface = random_noise(random_surface, mode="salt")

    # then store values only on the contour surface of structure_volume
    contour_strel = ball(1)
    eroded_structure = erosion(surface_mask, contour_strel)
    contour_region = surface_mask - eroded_structure

    for i in range(input_size[0]):
        random_surface[i, :, :] += i ** 5

    random_surface[contour_region == 0] = 0

    random_surface = np.interp(
        random_surface,
        (random_surface.min(), random_surface.max()),
        (value_limits[0], value_limits[1]),
    )

    return random_surface
