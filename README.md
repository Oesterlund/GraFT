# GraFT: Graph of Filaments over Time
<!--![alt text](https://github.com/Oesterlund/GraFT/blob/main/GraFt_logo.png) -->
<p align="center">
   <img src="https://github.com/Oesterlund/GraFT/blob/main/GraFT_logo.png" alt="alt text" width="500">
</p>


This tool is designed for identification and tracking of filamentous structures from 2D timeseries image data. Solely identification can also be performed if wanted.

The cytoskeleton supports essential cellular functions. In plant cells, microtubules guide cellulose synthesis and act as possible tensile stress sensors, whereas the actin cytoskeleton provides the basis for secretion and cytoplasmic streaming. While microtubule dynamics are evaluated by standardized image analysis tools, similar details of the actin cytoskeleton remains difficult to compute. This is largely due to the complex dynamics of actin filaments and bundles. Hence, new frameworks and tools to quantify and understand spatiotemporal behavior of the actin cytoskeleton is urgently needed to address cytoskeletal complexity in plant cells.
Here, we propose an automated, open-source and image-based algorithm named Graph of Filaments over Time, GraFT, that: (1) builds a network-based representation of segmented and enhanced filament-like structures based on pre-processed confocal images of plant cells; (2) identifies individual filaments by untangling the network representation. Here, individual filaments are defined based on the longest paths in a constrained depth first search, accounting for angles along the path. The untangling is modeled as a path cover, in which we allow overlap between filaments; (3) tracks filaments over time by solving a linear assignment problem using the information of spatial location of each individual filament. 

Importantly, GraFT can be used to determine and compare properties of filamentous structures in time-resolved image-data across different cell types as well as other filament-based systems; sparse or dense. Therefore, GraFT offers a substantial step towards an automated framework facilitating robust spatiotemporal studies of the actin cytoskeleton in cells. 

# Documentation

## Dependencies
The method was written in Python 3.8 and is dependent on the libraries mentioned below.
Current libraries with versions was used to build the code:
- networkx 2.8.4
- scikit-image 0.19.3
- scipy 1.9.3
- pandas 1.4.4
- numpy 1.23.5
- matplotlib 3.5.1
- simplification 0.6.2
- plotly 5.11.0
- astropy 5.1
- pickle 4.0

I recommend creating a new python environment using conda and activate it to install dependencies
```
conda create -n CDFS python
conda activate CDFS
```
And install the dependencies with either pip or conda
```
conda install NetworkX==2.8.4
conda install scikit-image 0.19.3
conda install scipy==1.9.3
conda install pandas==1.4.4
conda install numpy==1.23.5
conda install matplotlib==3.5.1
conda install simplification==0.6.2
conda install plotly==5.11.0
conda install astropy==5.1
```

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

## Real data
In the folder Article/ image data from the article is placed together with code used for data generation and data processing.
The data consists time-series data of Arabidopsis Thaliana, specifically:
- etiolated seedlings, imaged at three different locations along the stem
- seedlings treated with Latrunculin B etiolated and light-grown
- seedlings treated with virulence factors; DSF and flg22
In this folder you can also find the scripts used fort data generation and processing. If you want to use them, remember to change the folder names accordingly.


# References
The article for this algorithm is yet to be published. The article will be here to cite when it is out.
If you used this work, please remember to cite it.

# Licence
This project is licensed under the terms of the MIT license.
