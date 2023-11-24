# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import warnings

from astra.data.dataloader import get_loader
from monai.networks.nets import BasicUNet

from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter

import logging
import os
import sys
from datetime import datetime
import time
from collections import defaultdict

warnings.filterwarnings("ignore")


def trainer(args):
    data_root = "/Users/amithkamath/data/DLDP/ground_truth_small"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    outpath = os.path.join(args.exp_dir, args.exp)
    Path(outpath).mkdir(
        parents=True, exist_ok=True
    )  # create output directory to store model checkpoints
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")
    writer = SummaryWriter(
        outpath + "/" + date
    )  # create a date directory within the output directory for storing training logs

    # create training-validation data loaders
    list_eval_dirs = [
        os.path.join(data_root, "DLDP_") + str(i).zfill(3)
        for i in range(62, 80)
        if i not in [63, 65, 67, 77]  # missing data
    ]

    list_train_dirs = [
        os.path.join(data_root, "DLDP_") + str(i).zfill(3)
        for i in range(1, 62)
        if i != 40  # missing data
    ]

    data_paths = {
        "train": list_train_dirs,
        "val": list_eval_dirs,
    }

    train_loader, val_loader = get_loader(
        data_paths,
        train_bs=2,
        val_bs=1,
        train_num_samples_per_epoch=2 * 500,  # 500 iterations per epoch
        val_num_samples_per_epoch=1,
        num_works=4,
    )

    # create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256, 512, 32],
    ).to(device)
    print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))

    # create the loss function
    loss_function = torch.nn.L1Loss()

    # create the optimizer and the learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step_size, args.lr_gamma
    )

    # start a typical PyTorch training loop
    val_interval = 2  # doing validation every 2 epochs
    best_metric = -1
    best_metric_epoch = -1
    tic = time.time()
    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_idx, list_loader_output in enumerate(train_loader):
            step += 1
            optimizer.zero_grad()
            input_ = list_loader_output[0]
            target = list_loader_output[1:]

            # Forward
            output = model(input_.to(device))

            # Backward
            loss = loss_function(output, target.to(device))

            # Used for counting average loss of this epoch
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}, train_loss: {epoch_loss / step:.4f}", "\r", end="")

        scheduler.step()
        epoch_loss /= step
        writer.add_scalar("train_loss", epoch_loss, epoch + 1)
        print(
            f"epoch {epoch + 1} average loss: {epoch_loss:.4f} time elapsed: {(time.time()-tic)/60:.2f} mins"
        )

        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            outputs = defaultdict(list)
            targets = defaultdict(list)
            with torch.no_grad():
                val_ssim = list()
                val_loss = list()
                for val_data in val_loader:
                    input, target, mean, std, fname = (
                        val_data["kspace_masked_ifft"],
                        val_data["reconstruction_rss"],
                        val_data["mean"],
                        val_data["std"],
                        val_data["kspace_meta_dict"]["filename"],
                    )

                    # iterate through all slices:
                    slice_dim = (
                        1  # change this if another dimension is your slice dimension
                    )
                    num_slices = input.shape[slice_dim]
                    for i in range(num_slices):
                        inp = input[:, i, ...].unsqueeze(slice_dim)
                        tar = target[:, i, ...].unsqueeze(slice_dim)
                        output = model(inp.to(device))

                        vloss = loss_function(output, tar.to(device))
                        val_loss.append(vloss.item())

                        _std = std[0][i].item()
                        _mean = mean[0][i].item()
                        outputs[fname[0]].append(
                            output.data.cpu().numpy()[0][0] * _std + _mean
                        )
                        targets[fname[0]].append(tar.numpy()[0][0] * _std + _mean)

                # compute validation ssims
                for fname in outputs:
                    outputs[fname] = np.stack(outputs[fname])
                    targets[fname] = np.stack(targets[fname])
                    val_ssim.append(0.0)

                metric = np.mean(val_ssim)

                # save the best checkpoint so far
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(outpath, "unet_mri_reconstruction.pt"),
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean ssim: {:.4f} best mean ssim: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_ssim", metric, epoch + 1)

    print(
        f"training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


def main():
    parser = argparse.ArgumentParser()

    # training params
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="number of training epochs"
    )

    parser.add_argument(
        "--exp_dir",
        default="/Users/amithkamath/repo/astra/output",
        type=Path,
        help="output directory to save training logs",
    )

    parser.add_argument(
        "--exp",
        default="monai_testing",
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )

    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")

    parser.add_argument(
        "--lr_step_size",
        default=40,
        type=int,
        help="decay learning rate every lr_step_size epochs",
    )

    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="every lr_step_size epochs, decay learning rate by a factor of lr_gamma",
    )

    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="ridge regularization factor"
    )

    # model specific args
    parser.add_argument(
        "--drop_prob", default=0.0, type=float, help="dropout probability for U-Net"
    )

    args = parser.parse_args()
    trainer(args)


if __name__ == "__main__":
    main()
