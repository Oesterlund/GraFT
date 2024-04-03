import os
import filecmp
import hashlib
import shutil
import pickle

import numpy as np
import skimage.io as io
import pandas as pd
import pytest

from graft.run import create_all


def files_equal(file1_path, file2_path):
    """Return True iff both files are identical."""
    with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
        return hashlib.sha256(f1.read()).hexdigest() == hashlib.sha256(f2.read()).hexdigest()

@pytest.fixture(scope="module")  # fixture to setup and teardown test env
def test_env():
    # get the path of the directory that contains this script
    test_dir = os.path.dirname(os.path.realpath(__file__))

    image_path = os.path.join(test_dir, "..", "graft", "tiff", "timeseries.tif")
    test_output_dir = os.path.join(test_dir, "..", "test_output")
    expected_output_dir = os.path.join(test_dir, "expected_output")

    os.makedirs(test_output_dir, exist_ok=True)

    img_o = io.imread(os.path.abspath(image_path))
    create_all(pathsave=os.path.abspath(test_output_dir),
               img_o=img_o,
               maskDraw = np.ones((img_o.shape[1:3])),
               size=6, eps=200, thresh_top=0.5, sigma=1.0, small=50.0,
               angleA=140, overlap=4, max_cost=100,
               name_cell='in silico time')

    # Provide the paths that the test_regression function needs.
    yield test_output_dir, expected_output_dir

    # Teardown test environment
    shutil.rmtree(test_output_dir)

def test_regression_csv_files(test_env):
    output_dir, expected_dir = test_env

    # use pandas to compare CSV files with a tolerance for floating point errors
    for csv_fname in ("tracked_filaments_info.csv", "value_per_frame.csv"):
        generated_file = os.path.join(output_dir, csv_fname)
        expected_file = os.path.join(expected_dir, csv_fname)

        generated_df = pd.read_csv(generated_file)
        expected_df = pd.read_csv(expected_file)

        try:
            pd.testing.assert_frame_equal(generated_df, expected_df, check_dtype=False, atol=1e-5)
        except AssertionError as e:
            raise AssertionError(f"{csv_fname} does not match expected output. {e}")

def test_posL_pickle_file(test_env):
    output_dir, expected_dir = test_env

    fname = 'posL.gpickle'
    with open(os.path.join(output_dir, fname), 'rb') as f:
        generated_posL = pickle.load(f)
    with open(os.path.join(expected_dir, fname), 'rb') as f:
        expected_posL = pickle.load(f)

    assert len(generated_posL) == len(expected_posL), \
        "{fname}: number of frames does not match expected output."

    for frame_index, (generated_frame_pos, expected_frame_pos) in enumerate(zip(generated_posL, expected_posL)):
        assert np.array_equal(generated_frame_pos, expected_frame_pos), \
        f"Node positions don't match for frame {frame_index}."
