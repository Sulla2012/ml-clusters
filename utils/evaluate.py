import numpy as np

from astropy.io import fits
from astropy import wcs

import torch

from torch.utils.data.dataset import Subset


def box_cent(box):
    return [np.mean([box[0], box[2]]), np.mean([box[1], box[3]])]


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


def reduce_cat(cents, tol=3.0 / 60.0):
    unique_cents = []
    unique_cents.append(cents[0][0])
    for i in range(len(cents)):
        for j in range(len(cents[i])):
            if len(cents[i][j]) == 0:
                continue  # Filter cents with no detected clusters
            if not np.any(check_cat(cents[i][j], unique_cents, tol=tol)):
                unique_cents.append(cents[i][j])
    return unique_cents


def make_cat(model, dataset_test, img_path, device, thresh=0.5):
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


def make_cat_truth(dataset: Subset, img_path: str) -> np.array:
    """
    Make the catalog of the true clusters.

    Parameters
    ----------
    dataset : torch.utils.data.dataset.Subset
        Test data set with truths
    img_path : str
        Path to the truth images.

    Returns
    -------
    true_cents : np.Array
        Array of ra/dec for true clusters
    """
    true_cents = []
    for i in range(len(dataset)):
        # TODO: This should be functionized
        pixel_cents = []
        _, truth = dataset[i]
        wcs_img = fits.open(img_path.format(truth["image_id"][0]))
        w = wcs.WCS(wcs_img[0].header).dropaxis(
            2
        )  # wcs generates a dummy axis corresponding to frequency.
        for j in range(len(truth["boxes"])):
            pixel_cents.append(box_cent(truth["boxes"][j].cpu().numpy()))
        pixel_cents = np.array(pixel_cents, dtype=np.float64)
        sky_cents = w.wcs_pix2world(pixel_cents, 0)
        true_cents.append(sky_cents)

    return true_cents


def _get_real_clusters(
    true_cat: np.array, pred_cat: np.array, tol: float = 2.0 / 60
) -> list[int]:
    """
    For the purposes of comparison this function evalutes true clusters by looping thru
    the pred cat then the true cat, stopping the true cat loop whenever it finds
    a match. This can lead to a different issue where one true cluster amps to
    two predicted clusters, artificially boosting the purity. In testing
    the two methods produce the same purity, so I don't think this is a huge
    issue but one to be aware of.


    Parameters
    ----------
    true_cat : np.array
        Array of ra/dec for real, ground truth clusters
    pred_cat : np.array
        Array of ra/dec for predicted clusters
    tol : float, default = 1/60
        Tolerance to consider a pred and true cluster to be the same object.
        Units are degrees. TODO: you can run into issues where multiple
        pred clusters map to the same true cluster, in which case only the first
        found will be counted. This is in general the right behaviour but there
        can be very small edge cases where pred_a and true_a, pred_b and true_b
        are the same object but due to ordering pred_a is associated to ture_b
        because they are close on the sky but true_b is too far from true_a
        to match to true a. This is pretty niche tho

    Returns
    -------
    idx_real : list[int]
        Indexes of real clusters
    """

    idx_real = []
    for i in range(len(true_cat)):
        for j in range(len(pred_cat)):
            if np.linalg.norm(pred_cat[j] - true_cat[i]) <= tol:
                idx_real.append(j)
                continue
    return idx_real


def get_real_clusters(
    true_cat: np.array, pred_cat: np.array, tol: float = 2.0 / 60
) -> float:
    """
    Get the number of real clusters in a pred catalog as defined
    by a ground truth catalog.

    Parameters
    ----------
    true_cat : np.array
        Array of ra/dec for real, ground truth clusters
    pred_cat : np.array
        Array of ra/dec for predicted clusters
    tol : float, default = 1/60
        Tolerance to consider a pred and true cluster to be the same object.
        Units are degrees.

    Returns
    -------
    idx_real : list[int]
        Indexes of real clusters
    """

    idx_real = []
    for i in range(len(pred_cat)):
        for j in range(len(true_cat)):
            if np.linalg.norm(pred_cat[i] - true_cat[j]) <= tol:
                idx_real.append(j)
                continue
    return idx_real
