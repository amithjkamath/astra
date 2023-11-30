import os
import matplotlib.pyplot as plt
import k3d
import numpy as np
from k3d import matplotlib_color_maps

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

        data = np.load("streamlines_data.npz")
        plot = k3d.plot()
        plot += k3d.line(
            data["lines"],
            width=0.00007,
            color_range=[0, 0.5],
            shader="mesh",
            attribute=data["v"],
            color_map=matplotlib_color_maps.Inferno,
        )

        plot += k3d.mesh(
            data["vertices"],
            data["indices"],
            opacity=0.25,
            wireframe=True,
            color=0x0002,
        )
        plot.display()


if __name__ == "__main__":
    main()
