# -*- encoding: utf-8 -*-
import torch
from astra.model.cascaded_unet_monai import CascadedUNet
from astra.model.model import Model


def main():

    c3d_model = CascadedUNet(
        spatial_dims=3,
        in_channels=15,
        out_channels=1,
        channels_first=(16, 32, 64, 128, 256),
        channels_second=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        act="ReLU",
    )

    x = torch.randn(1, 15, 96, 96, 96, requires_grad=True)

    # Export the model
    c3d_model.eval()
    torch.onnx.export(
        c3d_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "cascaded_unet_monai.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    legacy_model = Model(
        in_ch=15,
        out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        list_ch_B=[-1, 32, 64, 128, 256, 512],
    )

    legacy_model.eval()
    torch.onnx.export(
        legacy_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "cascaded_unet_legacy.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
