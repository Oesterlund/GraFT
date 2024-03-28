import os
import filecmp
import hashlib
import shutil

import numpy as np
import skimage.io as io
import pytest

from graft.run import create_all


def files_equal(file1_path, file2_path):
    """Return True iff both files are identical."""
    with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
        return hashlib.sha256(f1.read()).hexdigest() == hashlib.sha256(f2.read()).hexdigest()

@pytest.fixture(scope="module")  # fixture to setup and teardown test env
def test_env():
    # Setup test environment
    image_path = "graft/tiff/timeseries.tif"
    test_output_dir = "test_output"
    os.makedirs(test_output_dir, exist_ok=True)

    img_o = io.imread(os.path.abspath(image_path))
    create_all(pathsave=os.path.abspath(test_output_dir),
               img_o=img_o,
               maskDraw = np.ones((img_o.shape[1:3])),
               size=6, eps=200, thresh_top=0.5, sigma=1.0, small=50.0,
               angleA=140, overlap=4, max_cost=100,
               name_cell='in silico time')
    
    yield test_output_dir  # Provide the test output directory to the test

    # Teardown test environment
    shutil.rmtree(test_output_dir)

# Test function to check output files
def test_regression(test_env):
    output_dir = test_env
    expected_dir = "expected_output"
    # Assuming expected_output contains the expected files
    
    # Compare file by file (Example for a specific file)
    assert files_equal(os.path.join(output_dir, "tracked_filaments_info.csv"),
                       os.path.join(expected_dir, "tracked_filaments_info.csv")), "tracked_filaments_info.csv differs"
    
    # TODO: Extend this to compare all expected files.
