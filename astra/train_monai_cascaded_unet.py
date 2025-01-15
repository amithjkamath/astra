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
from astra.model.CascadedUNet import CascadedUNet

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
    repo_root = "/home/akamath/Documents/astra/"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    out_path = os.path.join(args.exp_dir, args.exp)
    Path(out_path).mkdir(
        parents=True, exist_ok=True
    )  # create output directory to store model checkpoints
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")
    writer = SummaryWriter(
        out_path + "/" + date
    )  # create a date directory within the output directory for storing train logs

    # create train-validation data loaders
    data_root = "data/processed-to-train/ISAS_GBM_"

    list_eval_dirs = [
        os.path.join(repo_root, data_root) + str(i).zfill(3)
        for i in range(61, 70)
    ]

    list_train_dirs = [
        os.path.join(repo_root, data_root) + str(i).zfill(3)
        for i in range(1, 61)
    ]

    data_paths = {
        "train": list_train_dirs,
        "val": list_eval_dirs,
    }

    train_loader, val_loader = get_loader(
        data_paths,
        train_bs=2,
        val_bs=1,
        train_num_samples_per_epoch=1000,
        val_num_samples_per_epoch=1,
        num_workers=4,
    )

    # create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CascadedUNet(
        spatial_dims=3,
        in_channels=15,
        out_channels=1,
        channels_first=(16, 32, 64, 128, 256),
        channels_second=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        act="ReLU",
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

    # start a typical PyTorch train loop
    val_interval = 2  # doing validation every 2 epochs
    best_metric = 1e10
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
            target = list_loader_output[1]
            mask = list_loader_output[2]

            # Forward
            output = model(input_.to(device))

            # Backward
            output[mask == 0] = 0
            loss = loss_function(output, target.to(device))
            loss.backward()

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
            with torch.no_grad():
                val_loss_array = list()
                for batch_idx, list_loader_output in enumerate(val_loader):
                    input_ = list_loader_output[0]
                    target = list_loader_output[1]
                    mask = list_loader_output[2]

                    # Forward
                    output = model(input_.to(device))

                    # Backward
                    output[mask == 0] = 0

                    val_loss = loss_function(output, target.to(device))
                    val_loss_array.append(val_loss.item())

                metric = np.mean(val_loss_array)

                # save the best checkpoint so far
                if metric < best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            out_path + "/" + date, "cascaded_unet_dose_prediction.pt"
                        ),
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current MAE: {:.4f} best MAE: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mae", metric, epoch + 1)

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


def main():
    parser = argparse.ArgumentParser()

    # train params
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="number of train epochs"
    )

    parser.add_argument(
        "--exp_dir",
        default="/home/akamath/Documents/astra/output",
        type=Path,
        help="output directory to save train logs",
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
        default=2,
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
