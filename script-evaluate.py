import torch


from utils.nn import (
    ClusterDataset,
    get_transform,
    get_instance_frcnn_model,
)

from utils.evaluate import make_cat, reduce_cat, make_cat_truth, get_real_clusters


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


# def main():
parser = _make_parser()
args = parser.parse_args()

model_path = args.backbone_path + "/act-{}-frcnn-{}-tiles.pth".format(
    args.backbone, args.tile_type
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2
backbone = args.backbone
model = get_instance_frcnn_model(
    num_classes,
    backbone_path=args.backbone_path + "act-{}.pth".format(backbone),
    backbone_type=backbone,
)
model.to(device=device)

model.load_state_dict(torch.load(model_path))

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
# TODO: currently this relies on the seed being set the same for the training and
# evaluation which is really sketchy. I should probably just save the traingin
# and evaluation datasets separately.
torch.manual_seed(args.seed)
indices = torch.randperm(len(dataset)).tolist()

test_num = args.test_num
dataset = torch.utils.data.Subset(dataset, indices[:-test_num])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_num:])

img_path = (
    args.root + "/indv_freq_stamps/{:04}.fits"
)  # TODO: make work with other stamp types

cents = make_cat(
    model=model, dataset_test=dataset_test, img_path=img_path, device=device
)
pred_cat = reduce_cat(cents=cents)
true_cat = make_cat_truth(dataset=dataset_test, img_path=img_path)
true_cat = reduce_cat(true_cat, tol=1.0 / 60)

real_clusters = get_real_clusters(true_cat=true_cat, pred_cat=pred_cat)
n_real = len(real_clusters)
print("Purity: ", n_real / len(pred_cat))
print("Completeness: ", n_real / len(true_cat))

# if __name__ == "__main__":
#    main()
