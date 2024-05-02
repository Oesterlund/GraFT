from io import BytesIO
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import streamlit as st
import numpy as np
import tifffile as tiff
from skimage import io as skimage_io

from graft.main import (
    create_all, create_all_still, create_output_dirs, generate_default_mask)


# regex to find numbers in a string
INTEGER_RE = re.compile(r"(\d+)")

def natural_sort_key(s):
    """
    Extract a mixed numeric and non-numeric sorting key from strings
    for natural sort order, i.e. we want to make sure that
    'foo-2' preceedes 'foo-10'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(INTEGER_RE, str(s))]

def run():
    """
    Wrapper function to make this Streamlit app callable like a
    standalone application (via an entrypoint in pyproject.toml).
    """
    if 'streamlit' in sys.argv[0]:  # if we call "streamlit run" directly
        main()
    else:  # if we call 'graft-webapp' (the entrypoint defined in pyproject.toml)
        cmd = ["streamlit", "run", "src/graft/streamlit_app.py"]
        subprocess.run(cmd)

def main():
    # Main Page
    st.title('GraFT: Graph of Filaments over Time')
    uploaded_file = st.file_uploader("Upload TIFF file", type=['tif', 'tiff'])

    # Sidebar for configuration
    st.sidebar.title("Configuration")
    # all slider values: min, max, default
    sigma = st.sidebar.slider('Sigma', 0.5, 2.0, 1.0)
    small = st.sidebar.slider('Small', 30.0, 100.0, 50.0)
    angleA = st.sidebar.slider('Angle A', 100, 180, 140)
    overlap = st.sidebar.slider('Overlap', 1, 10, 4)
    max_cost = st.sidebar.slider('Max Cost', 50, 200, 100)
    size = st.sidebar.slider('Merge Radius (Size)', 1, 30, 6)
    eps = st.sidebar.slider('Epsilon', 1, 400, 200)
    thresh_top = st.sidebar.slider('thresh_top', 0.0, 1.0, 0.5)

    if uploaded_file is not None:
        bytes_data = BytesIO(uploaded_file.getvalue())
        
        try:
            # Use tifffile to read the TIFF file
            img_o = tiff.imread(bytes_data)

            if img_o.ndim not in (2,3):
                raise ValueError(f"Uploaded image is not a tiff still image or time-series. Expected image with 2 or 3 dimensions, got: {img_o.shape}.")

            mask = generate_default_mask(img_o.shape)

            # create temp directory structure for output files
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                create_output_dirs(str(output_dir))

                if img_o.ndim == 3:  # input image represents a time series
                    subdirs = ['n_graphs', 'circ_stat', 'mov', 'plots']
                    with st.spinner('Running analysis... Please wait'):
                        create_all(pathsave=str(output_dir), img_o=img_o, maskDraw=mask,
                                   size=size, eps=eps, thresh_top=thresh_top, sigma=sigma, small=small,
                                   angleA=angleA, overlap=overlap, max_cost=max_cost, name_cell='in silico time')
                        st.success("Analysis completed!")
                else:  # img_o.ndim == 2, i.e. input image is a still image
                    subdirs = ['n_graphs', 'circ_stat']
                    with st.spinner('Running analysis... Please wait'):
                        # `create_all_still` has the same parameters as `create_all`, except for `max_cost`
                        create_all_still(pathsave=str(output_dir), img_o=img_o, maskDraw=mask,
                                   size=size, eps=eps, thresh_top=thresh_top, sigma=sigma, small=small,
                                   angleA=angleA, overlap=overlap, name_cell='in silico still')
                        st.success("Analysis completed!")

                # Display images from all subdirectories using tabs
                tab_titles = [f"{subdir.replace('_', ' ').title()}" for subdir in subdirs]
                tabs = st.tabs(tab_titles)  # Create a tab for each subdirectory

                for tab, subdir in zip(tabs, subdirs):
                    with tab:
                        st.subheader(f"{subdir.replace('_', ' ').title()} Output")
                        subdir_path = Path(output_dir) / subdir
                        images = list(subdir_path.glob('*.png'))

                        # sort images naturally by filename
                        images_sorted = sorted(images, key=lambda x: natural_sort_key(x.name))
                        if images_sorted:
                            for image_path in images_sorted:
                                image = skimage_io.imread(str(image_path))
                                st.image(image, caption=f'{image_path.name}', use_column_width=True)
                        else:
                            st.write(f"No images found in {subdir}.")

        except Exception as e:
            st.error(str(e))
    else:
        st.warning("Please upload a TIFF file to proceed.")


if __name__ == "__main__":
    main()
