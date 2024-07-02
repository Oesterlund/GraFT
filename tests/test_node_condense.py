import pickle
import os
import numpy as np
import pytest
from scipy import sparse
from time import time
from memory_profiler import memory_usage
import sys

from graft import node_condense

"""
The purpose of this module is to compare the performance of different
implementations of the node_condense function and to ensure that they produce
identical results.
"""

def profile_memory(func, *args, **kwargs):
    def wrapper():
        return func(*args, **kwargs)
    mem_usage = memory_usage(wrapper, interval=0.1, retval=True)
    peak_memory = max(mem_usage[0])
    result = mem_usage[1]
    return peak_memory, result

def run_node_condense_test(node_condense_func):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    node_condense_capture_path = os.path.join(test_dir, "expected_output", "node_condense_sparse")
    test_files = os.listdir(node_condense_capture_path)
    input_files = [f for f in test_files if f.startswith('input_')]
    total_time_taken = 0
    total_memory_used = 0
    results = []

    # The input data is the same as `src/graft/data/timeseries.tif` but stored
    # as individual sparse matrices.
    for input_file in input_files:
        input_path = os.path.join(node_condense_capture_path, input_file)
        output_file = 'output_' + input_file.split('_')[1]
        output_path = os.path.join(node_condense_capture_path, output_file)

        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        with open(output_path, 'rb') as f:
            expected_output = pickle.load(f)

        # convert sparse matrices back to the "normal" ones that the
        # node_condense function expects.
        args = [arg.toarray() if isinstance(arg, sparse.csr_matrix) else arg for arg in input_data['args']]
        kwargs = {k: (v.toarray() if isinstance(v, sparse.csr_matrix) else v) for k, v in input_data['kwargs'].items()}
        expected_output = expected_output.toarray() if isinstance(expected_output, sparse.csr_matrix) else expected_output

        start_time = time()
        peak_memory, actual_output = profile_memory(node_condense_func, *args, **kwargs)
        end_time = time()

        time_taken = end_time - start_time
        total_time_taken += time_taken
        total_memory_used = max(total_memory_used, peak_memory)

        np.testing.assert_array_equal(actual_output, expected_output, err_msg="Output from node_condense does not match the expected output.")

        results.append({
            'input_file': input_file,
            'time_taken': time_taken,
            'memory_used': peak_memory
        })

    return total_time_taken, total_memory_used

def test_node_condense_performance():
    node_condense_versions = [
        ('node_condense_original', node_condense.node_condense_original),
        ('node_condense_faster_modularized', node_condense.node_condense_faster_modularized),
    ]

    for name, node_condense_func in node_condense_versions:
        total_time_taken, total_memory_used = run_node_condense_test(node_condense_func)
        print(f"\nTotal time taken by {name}: {total_time_taken:.4f} seconds, Peak memory used: {total_memory_used:.4f} MiB")

if __name__ == "__main__":
    pytest.main()
