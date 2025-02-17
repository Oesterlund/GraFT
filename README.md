# GraFT: Graph of Filaments over Time

<!--![alt text](https://github.com/Oesterlund/GraFT/blob/main/GraFt_logo.png) -->
<p align="center">
   <img src="https://github.com/Oesterlund/GraFT/blob/main/GraFT_logo.png" alt="alt text" width="500">
</p>


This tool is designed for identification and tracking of filamentous structures from 2D timeseries image data.

The cytoskeleton supports essential cellular functions. In plant cells, microtubules guide cellulose synthesis and act as possible tensile stress sensors, whereas the actin cytoskeleton provides the basis for secretion and cytoplasmic streaming. While microtubule dynamics are evaluated by standardized image analysis tools, similar details of the actin cytoskeleton remains difficult to compute. This is largely due to the complex dynamics of actin filaments and bundles. Hence, new frameworks and tools to quantify and understand spatiotemporal behavior of the actin cytoskeleton is urgently needed to address cytoskeletal complexity in plant cells.
Here, we propose an automated, open-source and image-based algorithm named Graph of Filaments over Time, GraFT, that: (1) builds a network-based representation of segmented and enhanced filament-like structures based on pre-processed confocal images of plant cells; (2) identifies individual filaments by untangling the network representation. Here, individual filaments are defined based on the longest paths in a constrained depth first search, accounting for angles along the path. The untangling is modeled as a path cover, in which we allow overlap between filaments; (3) tracks filaments over time by solving a linear assignment problem using the information of spatial location of each individual filament. 

## Overview
GraFT can be used to determine and compare properties of filamentous structures in time-resolved image-data across different cell types as well as other filament-based systems; sparse or dense. Therefore, GraFT offers a substantial step towards an automated framework facilitating robust spatiotemporal studies of the actin cytoskeleton in cells. 

The tool can perform identifcation on still image-data, and identification with tracking on time-series image data. The method can also be used to only preprocess and binarize, if the density of an image is only needed.

# Documentation

## Dependencies

This software was written using Python 3.8 and the following libraries: networkx, scikit-image, scipy, pandas, numpy, matplotlib, simplification, plotly, astropy and scikit-spatial (dependency versions are listed in `pyproject.toml`).

## Installation

I recommend creating a new Python environment using [conda](https://www.anaconda.com/download/success#miniconda) and activate it to install dependencies
```
conda create -n graft python=3.8
conda activate graft
```
And install the package and its dependencies using pip:

```
pip install -e .
```

Installation time does not take long, I recomend working with a GUI like [Anaconda-Navigator](https://docs.anaconda.com/free/navigator/index.html), which contains the IDE [Spyder](https://docs.anaconda.com/free/anaconda/ide-tutorials/spyder/).

## Usage

### GraFT web app

To run the web app locally, change to the directory where you cloned the repository and execute the command graft-webapp. The command line will display the server's address (e.g., http://localhost:8501). Typically, this process will automatically open a new browser tab pointing to that URL.

### GraFT command line interface

After you have changed to the directory where you cloned the repository, the GraFT CLI can be used like this on a time series image

```
graft-cli timeseries src/graft/data/timeseries.tif /tmp/graft_output
```

or like this on a still image:

```
graft-cli still src/graft/data/still_image.tif /tmp/graft_output
```

### GraFT Python library

GraFT can also be used as a library. For a concrete usage example, see `src/graft/example_run.py`.


# Workflow

## Data
Creating a mask is normally done using free hand drawing in FIJI, where the chosen area is filled with fill and divided by the value inside and background is removed by clear outside. This mask can then be saved in the folder with the data.
The image data is initially cleaned by rolling ball (radius=50) function with FIJI to remove unwanted background that would create a line when mask and image data is multiplied. In case one does not want to use FIJI, the python function skimage.restoration.rolling_ball can be used, and implemented like this:
```
import skimage restoration
background = restoration.rolling_ball(image,radius=50)
image_new = image - background
```
Instead of the cleaned image data with FIJI. This function is not the same as the FIJI rolling ball function, however quite similar. We found that the FIJI version performs best.
FIJI can be installed from the webpage [FIJI](https://imagej.net/software/fiji/)

You should now have a mask for your cell, and a cleaned version of your image data saved in your chosen directory.
You are now ready to use GraFT on your data!

## Getting started
In the folder GraFT you will find two scirpts and additional folders, download the main folder.
The two scripts utilsF.py and run.py is the code.  
The code contains a helper script, utilsF.py and a main script run.py calling the helper script. Please remember to change the line with the location of the utilsF script, to the one on your computer.
All import lines must be changed to the correct location on your computer, both for files, savepath and utilsF.
Run the function you wish, either create_all or create_all_still, and folders will be created in the directory location you defined with figures and datafile.
The folders contain the created figures and csv file of the outputs.

There are two main functions, one for timeseries image-data, and one for still image-data.  
The function for timeseries data is:
```
img_o = io.imread("/your_directory_path_here/your_image_file")
maskDraw = io.imread("/your_directory_path_here/your_mask_file")
create_all(pathsave = "/your_directory_path_here/",
           img_o=img_o,
           maskDraw=maskDraw,
           size=6,eps=200,thresh_top=0.5,sigma=1,small=50,angleA=140,overlap=4,max_cost=100,
           name_cell='your prefered name here')
```
Or for image-data containing one image only, use instead:
```
create_all_still(pathsave = "/your_directory_path_here/",
           img_o=img_o,
           maskDraw=maskDraw,
           size=6,eps=200,thresh_top=0.5,sigma=1,small=50,angleA=140,overlap=4,
           name_cell='your prefered name here')
```
Remember to change the parameters according to your image data.

## Versions this code has been tested on
This code has been tested and run on ubuntu 22.04 as well as macOS Ventura.
To install this on your own computer I recomend setting up a new Python environment for the dependencies, which should not take more than 10 min.
Runing the full example script run.py in the folder GraFT/GraFT/ takes less than 5 mins on ubuntu 22.04.

## Real data
On the data page Zenodo (DOI: 10.5281/zenodo.10476058) you can find the raw image-data used for data generation and data processing.
The data consists time-series data of Arabidopsis Thaliana, specifically:
- etiolated seedlings, imaged at three different locations along the stem
- seedlings treated with Latrunculin B etiolated and light-grown
- seedlings treated with virulence factors; DSF and flg22
In the folder GraFT/Article/code/ you can find the scripts used fort data generation and processing. If you want to use them, remember to change path and file names accordingly.


# References
The article for this algorithm is yet to be published. The article will be here to cite when it is out.
If you used this work, or found it helpful for your own work, please remember to cite it.

# Licence
This project is licensed under the terms of the MIT license.
