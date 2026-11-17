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

    for mtype in ["train", "test"]:

        # use our dataset and defined transformations
        cluster_dir = "{}_{}_freq_stamps".format(mtype, args.tile_type)
        mask_dir = "{}_{}_freq_masks".format(mtype, args.tile_type)
        dataset = ClusterDataset(
            args.root,
            get_transform(train=True),
            cluster_dir=cluster_dir,
            mask_dir=mask_dir
        )
    
        for i in range(len(dataset)):
            print(np.round(i/len(dataset), 3), end="\r")
            img, target = dataset.__getitem__(i)
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                idx = dataset.imgs[i].split(".")[0]
                cmd = f"rm {args.root}{cluster_dir}/{idx}.fits" 
                os.system(cmd)
                cmd = f"rm {args.root}{mask_dir}/{idx}_mask.npz"
                os.system(cmd)
    
if __name__ == "__main__":
    main()
