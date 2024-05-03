import pickle
import os
import numpy as np
import pytest
from scipy import sparse

from graft import utilsF


def test_node_condense():
    # get the path of the directory that contains this script
    test_dir = os.path.dirname(os.path.realpath(__file__))

    node_condense_capture_path = os.path.join(test_dir, "expected_output", "node_condense")

    test_files = os.listdir(node_condense_capture_path)

    # Filter out input files and associate them with their corresponding output files
    input_files = [f for f in test_files if f.startswith('input_')]
    for input_file in input_files:
        input_path = os.path.join(node_condense_capture_path, input_file)
        output_file = 'output_' + input_file.split('_')[1]
        output_path = os.path.join(node_condense_capture_path, output_file)

        # Load the input and output data
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        with open(output_path, 'rb') as f:
            expected_output = pickle.load(f)

        # Extract arguments and keyword arguments
        args = input_data['args']
        kwargs = input_data['kwargs']

        # Call the node_condense function
        actual_output = utilsF.node_condense(*args, **kwargs)

        # Assert that the actual output matches the expected output
        np.testing.assert_array_equal(actual_output, expected_output, err_msg="Output from node_condense does not match the expected output.")

