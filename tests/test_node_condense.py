import os
import pickle
from time import time

import numpy as np
import pytest
from scipy import sparse
from memory_profiler import memory_usage

from graft import utilsF


def test_node_condense_performance():
    # get the path of the directory that contains this script
    test_dir = os.path.dirname(os.path.realpath(__file__))

    node_condense_capture_path = os.path.join(test_dir, "expected_output", "node_condense_sparse")

    test_files = os.listdir(node_condense_capture_path)

    input_files = [f for f in test_files if f.startswith('input_')]
    results = []

    for input_file in input_files:
        input_path = os.path.join(node_condense_capture_path, input_file)
        output_file = 'output_' + input_file.split('_')[1]
        output_path = os.path.join(node_condense_capture_path, output_file)

        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        with open(output_path, 'rb') as f:
            expected_output = pickle.load(f)

        # convert sparse data back to dense arrays
        args = [arg.toarray() if isinstance(arg, sparse.csr_matrix) else arg for arg in input_data['args']]
        kwargs = {k: (v.toarray() if isinstance(v, sparse.csr_matrix) else v) for k, v in input_data['kwargs'].items()}
        expected_output = expected_output.toarray() if isinstance(expected_output, sparse.csr_matrix) else expected_output

        # measure time and memory for the original function
        start_time = time()
        mem_usage_before = memory_usage()[0]
        actual_output = utilsF.node_condense(*args, **kwargs)
        mem_usage_after = memory_usage()[0]
        end_time = time()

        time_taken = end_time - start_time
        memory_used = mem_usage_after - mem_usage_before

        # verify correctness
        np.testing.assert_array_equal(actual_output, expected_output, err_msg="Output from node_condense does not match the expected output.")

        results.append({
            'input_file': input_file,
            'time_taken': time_taken,
            'memory_used': memory_used
        })

    # print performance results
    for result in results:
        print(f"Test case {result['input_file']} - Time taken: {result['time_taken']:.4f} seconds, Memory used: {result['memory_used']:.4f} MiB")
