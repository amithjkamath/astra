import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
import os

from astra.data.utils import read_image_data, concatenate

plt.style.use("_mpl-gallery")


def main():

    data_root = "/Users/amithkamath/data/DLDP/"
    data_path = os.path.join(data_root, "ground_truth")

    list_test_dirs = [
        os.path.join(data_path, "DLDP_") + str(i).zfill(3) for i in range(81, 82)
    ]
    for test_dir in list_test_dirs:
        test_id = test_dir.split("/")[-1]
        dict_images = read_image_data(test_dir)

        """
        x, y, z = np.where(dict_images["Target"][0, ...])
        pdata = pyvista.PolyData(np.column_stack((x, y, z)))
        pdata["orig_sphere"] = np.ones(len(x))

        # create many spheres from the point cloud
        sphere = pyvista.Sphere(radius=0.25, phi_resolution=10, theta_resolution=10)
        pc = pdata.glyph(scale=False, geom=sphere, orient=False)
        pc.plot(cmap="Reds")
        """

        p = pv.Plotter()

        mesh = pv.Cube(
            center=(64, 64, 64), x_length=128.0, y_length=128.0, z_length=128.0,
        )
        p.add_mesh(mesh, color="red", style="wireframe", line_width=2)

        for structure in ["Target"]:
            p.add_volume(
                np.multiply(dict_images[structure][0, ...], 255), cmap="jet", shade=True
            )

        for structure in ["Eye_L", "Eye_R", "BrainStem"]:

            p.add_volume(
                np.multiply(dict_images[structure][0, ...], 255),
                shade=True,
                cmap="cool",
            )
        p.show()


if __name__ == "__main__":
    main()
