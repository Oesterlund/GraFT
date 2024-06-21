import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile
import shutil

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

def add_results_download_button(output_dir, md5_sum, parameters):
    # generate log file
    log_content = (
        f"MD5 sum of input file: {md5_sum}\n"
        "Parameters:\n"
    )
    for param, value in parameters.items():
        log_content += f"\t{param}: {value}\n"

    log_file_path = Path(output_dir) / "analysis_log.txt"
    with open(log_file_path, 'w') as log_file:
        log_file.write(log_content)

    # create a zip file of the results
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for folder_name, subfolders, filenames in os.walk(output_dir):
            for filename in filenames:
                file_path = Path(folder_name) / filename
                zipf.write(file_path, arcname=file_path.relative_to(output_dir))
    zip_buffer.seek(0)

    # add download button
    st.download_button(
        label="Download Results as ZIP",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )

def get_md5sum(uploaded_file):
    md5 = hashlib.md5()
    md5.update(uploaded_file.getvalue())
    return md5.hexdigest()

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
        st.session_state['output_dir'] = None
        st.session_state['md5_sum'] = None
        st.session_state['params'] = None

    def reset_session_state():
        if st.session_state['output_dir'] and Path(st.session_state['output_dir']).exists():
            shutil.rmtree(st.session_state['output_dir'])
        st.session_state['analysis_results'] = None
        st.session_state['output_dir'] = None
        st.session_state['md5_sum'] = None
        st.session_state['params'] = None

    def run_analysis(bytes_data, params):
        try:
            # Use tifffile to read the TIFF file
            img_o = tiff.imread(bytes_data)

            if img_o.ndim not in (2,3):
                raise ValueError(f"Uploaded image is not a tiff still image or time-series. Expected image with 2 or 3 dimensions, got: {img_o.shape}.")

            mask = generate_default_mask(img_o.shape)

            # create a persistent temp directory structure for output files
            temp_dir = tempfile.mkdtemp()
            output_dir = Path(temp_dir)
            create_output_dirs(str(output_dir))

            if img_o.ndim == 3:  # input image represents a time series
                subdirs = ['n_graphs', 'circ_stat', 'mov', 'plots']
                with st.spinner('Running analysis... Please wait'):
                    start_time = time.time()
                    create_all(pathsave=str(output_dir), img_o=img_o, maskDraw=mask,
                        size=params["Merge Radius (Size)"], eps=params["Epsilon"],
                        thresh_top=params["Thresh Top"], sigma=params["Sigma"],
                        small=params["Small"], angleA=params["Angle A"],
                        overlap=params["Overlap"], max_cost=params["Max Cost"],
                        name_cell='in silico time')
                    st.success(f"Analysis completed! Time taken: {time.time() - start_time:.2f} seconds.")
                    st.session_state['analysis_results'] = True
                    st.session_state['output_dir'] = output_dir
                    st.session_state['md5_sum'] = get_md5sum(uploaded_file)
                    st.session_state['params'] = params

            else:  # img_o.ndim == 2, i.e. input image is a still image
                subdirs = ['n_graphs', 'circ_stat']
                with st.spinner('Running analysis... Please wait'):
                    start_time = time.time()
                    create_all(pathsave=str(output_dir), img_o=img_o, maskDraw=mask,
                        size=params["Merge Radius (Size)"], eps=params["Epsilon"],
                        thresh_top=params["Thresh Top"], sigma=params["Sigma"],
                        small=params["Small"], angleA=params["Angle A"],
                        overlap=params["Overlap"], name_cell='in silico still')
                    st.success(f"Analysis completed! Time taken: {time.time() - start_time:.2f} seconds.")
                    st.session_state['analysis_results'] = True
                    st.session_state['output_dir'] = output_dir
                    st.session_state['md5_sum'] = get_md5sum(uploaded_file)
                    st.session_state['params'] = params

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

    # Main Page
    st.title('GraFT: Graph of Filaments over Time')
    uploaded_file = st.file_uploader("Upload TIFF file", type=['tif', 'tiff'], on_change=reset_session_state)

    # sidebar for configuration
    st.sidebar.title("Configuration")
    params = {  # all slider values: min, max, default
        "Sigma": st.sidebar.slider('Sigma', 0.5, 2.0, 1.0, on_change=reset_session_state),
        "Small": st.sidebar.slider('Small', 30.0, 100.0, 50.0, on_change=reset_session_state),
        "Angle A": st.sidebar.slider('Angle A', 100, 180, 140, on_change=reset_session_state),
        "Overlap": st.sidebar.slider('Overlap', 1, 10, 4, on_change=reset_session_state),
        "Max Cost": st.sidebar.slider('Max Cost', 50, 200, 100, on_change=reset_session_state),
        "Merge Radius (Size)": st.sidebar.slider('Merge Radius (Size)', 1, 30, 6, on_change=reset_session_state),
        "Epsilon": st.sidebar.slider('Epsilon', 1, 400, 200, on_change=reset_session_state),
        "Thresh Top": st.sidebar.slider('thresh_top', 0.0, 1.0, 0.5, on_change=reset_session_state)
    }

    if uploaded_file is not None:
        st.session_state['params'] = params
        if st.button('Run Analysis'):
            run_analysis(BytesIO(uploaded_file.getvalue()), params)

    if st.session_state['analysis_results']:
        add_results_download_button(st.session_state['output_dir'], st.session_state['md5_sum'], st.session_state['params'])

if __name__ == "__main__":
    main()
