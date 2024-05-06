import pickle
import os
import numpy as np
import pytest
from scipy import sparse

from graft import utilsF



def test_node_condense_sparse():
    # get the path of the directory that contains this script
    test_dir = os.path.dirname(os.path.realpath(__file__))

    node_condense_capture_path = os.path.join(test_dir, "expected_output", "node_condense_sparse")

    test_files = os.listdir(node_condense_capture_path)

    input_files = [f for f in test_files if f.startswith('input_')]
    for input_file in input_files:
        input_path = os.path.join(node_condense_capture_path, input_file)
        output_file = 'output_' + input_file.split('_')[1]
        output_path = os.path.join(node_condense_capture_path, output_file)

        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        with open(output_path, 'rb') as f:
            expected_output = pickle.load(f)

        # Convert sparse data back to dense arrays
        args = [arg.toarray() if isinstance(arg, sparse.csr_matrix) else arg for arg in input_data['args']]
        kwargs = {k: (v.toarray() if isinstance(v, sparse.csr_matrix) else v) for k, v in input_data['kwargs'].items()}
        expected_output = expected_output.toarray() if isinstance(expected_output, sparse.csr_matrix) else expected_output

        actual_output = utilsF.node_condense(*args, **kwargs)

        np.testing.assert_array_equal(actual_output, expected_output, err_msg="Output from node_condense does not match the expected output.")
