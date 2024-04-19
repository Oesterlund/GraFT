"""
This module contains regression tests for the function `graft.main.create_all_still`.

`graft.main.create_all_still` has huge overlaps with `graft.main.create_all`,
we need to add regression tests before we can refactor both of them.
"""

import os
import shutil
import pickle

import networkx as nx
import numpy as np
import skimage.io as io
import pytest

from graft.main import create_all_still
from tests import assert_csv_files_equal, compare_images


@pytest.fixture(scope="module")  # fixture to setup and teardown test env
def test_env():
    # get the path of the directory that contains this script
    test_dir = os.path.dirname(os.path.realpath(__file__))

    image_path = os.path.join(test_dir, "..", "graft", "tiff", "still_image.tif")
    test_output_dir = os.path.join(test_dir, "..", "test_output", "create_all_still")
    expected_output_dir = os.path.join(test_dir, "expected_output", "create_all_still")

    os.makedirs(test_output_dir, exist_ok=True)

    img = io.imread(os.path.abspath(image_path))

    create_all_still(pathsave=test_output_dir,
               img_o=img,
               maskDraw=np.ones(img.shape),
               size=6, eps=200, thresh_top=0.5, sigma=1.0, small=50.0,
               angleA=140, overlap=4, name_cell='in silico still')


    # Provide the paths that the test_regression function needs.
    yield test_output_dir, expected_output_dir

    # Teardown test environment
    shutil.rmtree(test_output_dir)


def test_regression_csv_files(test_env):
    output_dir, expected_dir = test_env

    # NOTE: the "same" file is called "tracked_filaments_info.csv" in create_all (traced -> tracked).
    csv_fname = "traced_filaments_info.csv"
    generated_file = os.path.join(output_dir, csv_fname)
    expected_file = os.path.join(expected_dir, csv_fname)
    # ~ breakpoint()
    assert_csv_files_equal(generated_file, expected_file)


def test_plot_images(test_env):
    """
    Test that all generated images are structurally similar to the
    expected output images.
    
    TODO / FIXME: replace this stopgap test with a real test that
    compares the underlying data structures instead of the generated images.
    """
    output_dir, expected_dir = test_env

    for plot_dir in ('circ_stat', 'n_graphs'):
        output_plot_dir = os.path.join(output_dir, plot_dir)
        expected_plot_dir = os.path.join(expected_dir, plot_dir)

        for plot_fname in os.listdir(expected_plot_dir):
            if plot_fname.endswith('.png'):
                generated_plot_path = os.path.join(output_plot_dir, plot_fname)
                expected_plot_path = os.path.join(expected_plot_dir, plot_fname)

                assert compare_images(generated_plot_path, expected_plot_path, method='ssim', tolerance=0.95), \
                    f"{plot_fname} does not match the expected output."
