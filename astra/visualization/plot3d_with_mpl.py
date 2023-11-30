import matplotlib.pyplot as plt
import numpy as np
import os

from astra.data.utils import read_image_data, concatenate

plt.style.use("_mpl-gallery")


def main():

    data_root = "/Users/amithkamath/data/DLDP/"
    data_path = os.path.join(data_root, "ground_truth_small")

    list_test_dirs = [
        os.path.join(data_path, "DLDP_") + str(i).zfill(3) for i in range(81, 82)
    ]
    for test_dir in list_test_dirs:
        test_id = test_dir.split("/")[-1]
        dict_images = read_image_data(test_dir)

        """
        # Prepare some coordinates
        x, y, z = np.indices((8, 8, 8))

        # Draw cuboids in the top left and bottom right corners
        cube1 = (x < 3) & (y < 3) & (z < 3)
        cube2 = (x >= 5) & (y >= 5) & (z >= 5)

        # Combine the objects into a single boolean array
        voxelarray = cube1 | cube2
        """
        voxelarray = (
            dict_images["Target"][0, ...]
            | dict_images["Eye_L"][0, ...]
            | dict_images["Eye_R"][0, ...]
        )

        # Plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(azim=30)

        ax.voxels(voxelarray, edgecolor="k")

        ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

        plt.show()


if __name__ == "__main__":
    main()
