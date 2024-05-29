import pickle
import os
import numpy as np
import pytest
from scipy import sparse
from time import time
from memory_profiler import memory_usage

# Import both versions of node_condense for comparison
from graft import utilsF

def profile_memory(func, *args, **kwargs):
    def wrapper():
        return func(*args, **kwargs)
    mem_usage = memory_usage(wrapper, interval=0.1, retval=True)
    peak_memory = max(mem_usage[0])
    result = mem_usage[1]
    return peak_memory, result

def run_node_condense_test(node_condense_func):
    # Get the path of the directory that contains this script
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

        # Convert sparse data back to dense arrays
        args = [arg.toarray() if isinstance(arg, sparse.csr_matrix) else arg for arg in input_data['args']]
        kwargs = {k: (v.toarray() if isinstance(v, sparse.csr_matrix) else v) for k, v in input_data['kwargs'].items()}
        expected_output = expected_output.toarray() if isinstance(expected_output, sparse.csr_matrix) else expected_output

        # Measure time and memory for the given function
        start_time = time()
        peak_memory, actual_output = profile_memory(node_condense_func, *args, **kwargs)
        end_time = time()

        time_taken = end_time - start_time
        memory_used = peak_memory  # Memory in MiB

        # Verify correctness
        np.testing.assert_array_equal(actual_output, expected_output, err_msg="Output from node_condense does not match the expected output.")

        results.append({
            'input_file': input_file,
            'time_taken': time_taken,
            'memory_used': memory_used
        })

    return results

def test_node_condense_performance():
    # List of node_condense function versions to test
    node_condense_versions = [
        ('node_condense', utilsF.node_condense),
        # Add other versions here, e.g., ('optimized_node_condense', optimized_node_condense)
    ]

    for name, node_condense_func in node_condense_versions:
        results = run_node_condense_test(node_condense_func)
        # Print performance results for each version
        for result in results:
            print(f"Test case {result['input_file']} with {name} - Time taken: {result['time_taken']:.4f} seconds, Memory used: {result['memory_used']:.4f} MiB")
