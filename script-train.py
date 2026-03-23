import torch

import os

from visionutils.utils import collate_fn
from visionutils.engine import train_one_epoch, evaluate

from utils.nn import (
    ClusterDataset,
    get_transform,
    get_instance_frcnn_model,
)

import argparse as argp


def _make_parser() -> argp.ArgumentParser:
    parser = argp.ArgumentParser(
        description="Train a neural network on a set of images of galaxy clusters"
    )
    parser.add_argument("root", help="Path to the dataset of images")
    parser.add_argument(
        "--tile_type",
        "-tt",
        type=str,
        default="indv",
        help="What type of images to train on.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=1,
        help="Seed to use for splitting train/test.",
    )
    parser.add_argument(
        "--test_num",
        "-tn",
        type=int,
        default=200,
        help="Number of training images to reserve.",
    )
    parser.add_argument(
        "--backbone",
        "-bb",
        type=str,
        default="mobilenet",
        help="Backbone class to use.",
    )
    parser.add_argument(
        "--backbone_path",
        "-bp",
        type=str,
        default="/mnt/welch/USERS/jorlo/ml-clusters/models/torch-act/",
        help="Path to backbone save location",
    )  # TODO: maybe this should be required
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        default=10,
        help="Number of epochs to train for.",
    )
    return parser


def main():
    parser = _make_parser()
    args = parser.parse_args()

    # use our dataset and defined transformations
    dataset = ClusterDataset(
        args.root,
        get_transform(train=True),
        cluster_dir="{}_freq_stamps".format(args.tile_type),
        mask_dir="{}_freq_masks".format(args.tile_type),
    )
    dataset_test = ClusterDataset(
        args.root,
        get_transform(train=False),
        cluster_dir="{}_freq_stamps".format(args.tile_type),
        mask_dir="{}_freq_masks".format(args.tile_type),
    )

    # split the dataset in train and test set
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset)).tolist()

    test_num = args.test_num
    dataset = torch.utils.data.Subset(dataset, indices[:-test_num])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_num:])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # our dataset has two classes only - background and person
    num_classes = 2
    backbone = args.backbone
    backbone_path = args.backbone_path
    # get the model using our helper function
    model = get_instance_frcnn_model(
        num_classes,
        backbone_path=backbone_path + "act-{}.pth".format(backbone),
        backbone_type=backbone,
    )

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )  # TODO: make parameters command line adjustable

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = args.num_epochs
    model_path = "/mnt/welch/USERS/jorlo/ml-clusters/models/torch-act/act-{}-frcnn-{}-tiles.pth".format(
        backbone, args.tile_type
    )  # TODO: fix this path
    load_exiting_weights = True
    if load_exiting_weights and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
