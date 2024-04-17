import hashlib

import numpy as np
import pandas as pd
from skimage.color import rgb2gray
import skimage.io as io
from skimage.metrics import structural_similarity as ssim


def files_equal(file1_path, file2_path):
    """Return True iff both files are identical."""
    with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
        return hashlib.sha256(f1.read()).hexdigest() == hashlib.sha256(f2.read()).hexdigest()

def assert_csv_files_equal(csv_file1_path, csv_file2_path, tolerance=1e-5):
    """Check if two CSV files are equal with a tolerance for floating point errors."""
    csv1_df = pd.read_csv(csv_file1_path)
    csv2_df = pd.read_csv(csv_file2_path)

    try:
        pd.testing.assert_frame_equal(csv1_df, csv2_df, check_dtype=False, atol=tolerance)
    except AssertionError as e:
        raise AssertionError(f"CSV files ({csv_file1_path}, {csv_file2_path}) are not equal. {e}")

def compare_images(image_path_1, image_path_2, method='ssim', tolerance=0.99):
    """
    Compare two images using either direct numpy comparison or SSIM.
    :param image_path_1: Path to the first image.
    :param image_path_2: Path to the second image.
    :param method: 'ssim' for Structural Similarity Index or 'direct' for direct numpy comparison.
    :param tolerance: Tolerance threshold for SSIM. Images are considered similar if SSIM >= tolerance.
    :return: True if images are considered equal, False otherwise.
    """
    img1 = io.imread(image_path_1)
    img2 = io.imread(image_path_2)

    # remove alpha channel if images have 4 channels (RGBA)
    if img1.shape[-1] == 4:
        img1 = img1[..., :3]
    if img2.shape[-1] == 4:
        img2 = img2[..., :3]

    if method == 'direct':
        return np.array_equal(img1, img2)
    elif method == 'ssim':
        # convert images to grayscale to compute SSIM
        img1_gray = rgb2gray(img1)
        img2_gray = rgb2gray(img2)

        ssim_index = ssim(img1_gray, img2_gray)
        return ssim_index >= tolerance
