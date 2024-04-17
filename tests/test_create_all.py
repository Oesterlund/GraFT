"""
This module contains regression tests for the function `graft.main.create_all`.
"""

import os
import shutil
import pickle

import networkx as nx
import numpy as np
import skimage.io as io
import pytest

from graft.main import create_all
from tests import assert_csv_files_equal, compare_images


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

    for csv_fname in ("tracked_filaments_info.csv", "value_per_frame.csv"):
        generated_file = os.path.join(output_dir, csv_fname)
        expected_file = os.path.join(expected_dir, csv_fname)
        assert_csv_files_equal(generated_file, expected_file)

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

def compare_edge_attributes(generated_edges, expected_edges):
    for ((key1, attr1), (key2, attr2)) in zip(generated_edges, expected_edges):
        assert key1 == key2, "Edge keys do not match."
        assert attr1 == attr2, "Edge attributes do not match for key {}".format(key1)

def test_tagged_graph_pickle_file(test_env):
    output_dir, expected_dir = test_env

    with open(os.path.join(output_dir, 'tagged_graph.gpickle'), 'rb') as f:
        generated_graphs = pickle.load(f)
    with open(os.path.join(expected_dir, 'tagged_graph.gpickle'), 'rb') as f:
        expected_graphs = pickle.load(f)

    assert len(generated_graphs) == len(expected_graphs), "Number of graphs does not match."

    for generated_graph, expected_graph in zip(generated_graphs, expected_graphs):
        assert set(generated_graph.nodes()) == set(expected_graph.nodes()), "Graph nodes do not match."
        assert nx.number_of_edges(generated_graph) == nx.number_of_edges(expected_graph), "Number of edges do not match."

        # Compare all edges and attributes for each pair of nodes
        for (u, v) in generated_graph.edges():
            generated_edges = sorted(generated_graph.get_edge_data(u, v).items(), key=lambda x: x[0])
            expected_edges = sorted(expected_graph.get_edge_data(u, v).items(), key=lambda x: x[0])

            assert len(generated_edges) == len(expected_edges), f"Edges between nodes {u} and {v} do not match in count."
            compare_edge_attributes(generated_edges, expected_edges)

def test_plot_images(test_env):
    """
    Test that all generated images are structurally similar to the
    expected output images.
    
    TODO / FIXME: replace this stopgap test with a real test that
    compares the underlying data structures instead of the generated images.
    """
    output_dir, expected_dir = test_env

    for plot_dir in ('circ_stat', 'mov', 'n_graphs', 'plots'):
        output_plot_dir = os.path.join(output_dir, plot_dir)
        expected_plot_dir = os.path.join(expected_dir, plot_dir)

        for plot_fname in os.listdir(expected_plot_dir):
            if plot_fname.endswith('.png'):
                generated_plot_path = os.path.join(output_plot_dir, plot_fname)
                expected_plot_path = os.path.join(expected_plot_dir, plot_fname)

                assert compare_images(generated_plot_path, expected_plot_path, method='ssim', tolerance=0.95), \
                    f"{plot_fname} does not match the expected output."
