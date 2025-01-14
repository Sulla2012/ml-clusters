import numpy as np

from astropy.io import fits
from astropy import wcs

import torch


def box_cent(box):
    return [np.mean([box[0], box[1]]), np.mean([box[2], box[3]])]


def center_dist(pred, truth):
    pred_cent = box_cent(pred)
    truth_cent = box_cent(truth)

    return np.sqrt(
        np.abs(pred_cent[0] - truth_cent[0]) ** 2
        + np.abs(pred_cent[1] - truth_cent[1]) ** 2
    )


def check_cat(cent, cat, tol=3.0 / 60.0):
    cat = np.array(cat)
    return np.array(
        [np.sqrt((cent[0] - cat[:, 0]) ** 2 + (cent[1] - cat[:, 1]) ** 2) < tol]
    )


def reduce_cat(cents):
    unique_cents = []
    unique_cents.append(cents[0][0])
    for i in range(len(cents)):
        for j in range(len(cents[i])):
            if len(cents[i][j]) == 0:
                continue  # Filter cents with no detected clusters
            if not np.any(check_cat(cents[i][j], unique_cents)):
                unique_cents.append(cents[i][j])
    return unique_cents


def make_catalog(model, dataset_test, img_path, thresh=0.5):
    model.eval()
    cents = []
    for i in range(len(dataset_test)):
        pixel_cents = []
        img, truth = dataset_test[i]
        wcs_img = fits.open(img_path.format(truth["image_id"][0]))
        w = wcs.WCS(wcs_img[0].header).dropaxis(
            2
        )  # wcs generates a dummy axis corresponding to frequency.

        with torch.no_grad():
            prediction = model([img.to(device)])[
                0
            ]  # Only one input img make a dummy axis
        for j in range(len(prediction["boxes"])):
            if prediction["scores"][j] >= thresh:
                pixel_cents.append(box_cent(prediction["boxes"][j].cpu().numpy()))
        pixel_cents = np.array(pixel_cents, dtype=np.float64)
        sky_cents = w.wcs_pix2world(pixel_cents, 0)
        cents.append(sky_cents)
    return cents


def eval_model(model, dataset_test, thresh=0.5):
    model.eval()
    # First evaluate completenes
    detected = np.zeros(len(dataset_test))
    purity = 0
    for i in range(len(dataset_test)):
        img, truth = dataset_test[i]
        with torch.no_grad():
            prediction = model([img.to(device)])[
                0
            ]  # Only one input img make a dummy axis
        for j in range(len(prediction["boxes"])):
            if detected[i] == 1:
                continue  # If we already found this cluster, continue. Probably better control for this
            if prediction["scores"][j] >= thresh:
                for k in range(
                    len(truth["boxes"])
                ):  # Check each true cluter within image.
                    if (
                        center_dist(
                            prediction["boxes"][j].cpu().numpy(),
                            truth["boxes"][k].cpu().numpy(),
                        )
                        < 2
                    ):
                        detected[i] = 1
                        continue

    purity = 0
    for i in range(len(dataset_test)):
        img, truth = dataset_test[i]
    return detected
