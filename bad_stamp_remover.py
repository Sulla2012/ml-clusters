import os

import numpy as np

from utils.nn import (
    ClusterDataset,
    get_transform,
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

    for i in range(len(dataset)):
        print(np.round(i/len(dataset), 3), end="\r")
        img, target = dataset.__getitem__(i)
        boxes = target["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            idx = dataset.imgs[i].split(".")[0]
            cmd = "rm /mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/indv_freq_masks/{}_mask.npz".format(idx)
            os.system(cmd)
            cmd = "rm /mnt/welch/USERS/jorlo/ml-clusters/websky_tiles/indv_freq_stamps/{}.fits".format(idx)
            os.system(cmd)

if __name__ == "__main__":
    main()
