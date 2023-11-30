import numpy as np
import torch
import warnings
import wandb
from datetime import datetime

from astra.data.dataloader import get_loader
from monai.networks.nets import BasicUNet
from monai.utils import set_determinism

from pathlib import Path

import logging
import os
import sys
import time

warnings.filterwarnings("ignore")


def trainer(config):
    data_root = "/Users/amithkamath/data/DLDP/ground_truth_small"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    outpath = os.path.join(config["exp_directory"], config["exp_name"])
    Path(outpath).mkdir(
        parents=True, exist_ok=True
    )  # create output directory to store model checkpoints
    os.makedirs(os.path.join(outpath + "/" + config["date"]), exist_ok=True)

    # Set deterministic training for reproducibility
    set_determinism(seed=config["seed"])

    # create train-validation data loaders
    list_train_dirs = [
        os.path.join(data_root, "DLDP_") + str(i).zfill(3)
        for i in range(1, 61)
        if i != 40
    ]

    list_val_dirs = [
        os.path.join(data_root, "DLDP_") + str(i).zfill(3)
        for i in range(61, 81)
        if i not in [63, 65, 67, 77]  # missing data
    ]

    data_paths = {
        "train": list_train_dirs,
        "val": list_val_dirs,
    }

    train_loader, val_loader = get_loader(
        data_paths,
        train_bs=config["train_batch_size"],
        val_bs=config["val_batch_size"],
        train_num_samples_per_epoch=config["train_batch_size"] * len(list_train_dirs),
        val_num_samples_per_epoch=config["val_batch_size"] * len(list_val_dirs),
        num_workers=config["num_workers"],
    )

    # create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(**config["model_params"]).to(device)
    wandb.watch(model, log_freq=100)
    print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))

    # create the loss function
    loss_function = torch.nn.L1Loss()

    # create the optimizer and the learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config["lr_decay_step_size"], config["lr_gamma"],
    )

    # start a typical PyTorch train loop
    val_interval = config["val_interval"]  # doing validation every 2 epochs
    best_metric = 1e10
    best_metric_epoch = -1
    tic = time.time()
    for epoch in range(config["num_epochs"]):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config['num_epochs']}")
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
            wandb.log({"train/loss": loss.item()})
            print(f"{step}, train_loss: {epoch_loss / step:.4f}", "\r", end="")

        scheduler.step()
        epoch_loss /= step
        wandb.log({"train/loss_epoch": epoch_loss})
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
                wandb.log({"val/metric": metric})

                # save the best checkpoint so far
                if metric < best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            outpath + "/" + config["date"], "unet_dose_prediction.pt"
                        ),
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current MAE: {:.4f} best MAE: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    wandb.log(
        {"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch,}
    )
    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )


def main():

    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")

    config = {
        # experiment settings
        "exp_directory": "/Users/amithkamath/repo/astra/output",
        "exp_name": "basic-unet-test",
        "date": date,
        # data
        "cache_rate": 1.0,
        "num_workers": 4,
        "seed": 1,
        # train settings
        "num_epochs": 5,
        "val_interval": 2,  # check validation score after n epochs
        "train_batch_size": 2,
        "val_batch_size": 1,
        "learning_rate": 1e-4,
        "lr_scheduler": "stepLR",  # just to keep track
        "lr_decay_step_size": 10,
        "lr_gamma": 0.5,
        "weight_decay": 0.0,
        # Unet model
        "model_type": "BasicUNet",  # just to keep track
        "model_params": dict(
            spatial_dims=3,
            in_channels=15,
            out_channels=1,
            features=[32, 64, 128, 256, 512, 32],
        ),
    }

    project_name = date + "-seed-" + str(config["seed"])
    wandb.init(
        project="BasicUNet-small-dose-prediction", name=project_name, config=config,
    )
    trainer(config)
    wandb.finish()


if __name__ == "__main__":
    main()
