# GraFT: Graph of Filaments over Time
![alt text](https://github.com/Oesterlund/GraFT/blob/main/nicegraph2.png)

This tool is designed for identification and tracking of filamentous structures from 2D timeseries image data.

The cytoskeleton supports essential cellular functions. In plant cells, microtubules guide cellulose synthesis and act as possible tensile stress sensors, whereas the actin cytoskeleton provides the basis for secretion and cytoplasmic streaming. While microtubule dynamics are evaluated by standardized image analysis tools, similar details of the actin cytoskeleton remains difficult to compute. This is largely due to the complex dynamics of actin filaments and bundles. Hence, new frameworks and tools to quantify and understand spatiotemporal behavior of the actin cytoskeleton is urgently needed to address cytoskeletal complexity in plant cells.
Here, we propose an automated, open-source and image-based algorithm named Graph of Filaments over Time, GraFT, that: (1) builds a network-based representation of segmented and enhanced filament-like structures based on pre-processed confocal images of plant cells; (2) identifies individual filaments by untangling the network representation. Here, individual filaments are defined based on the longest paths in a constrained depth first search, accounting for angles along the path. The untangling is modeled as a path cover, in which we allow overlap between filaments; (3) tracks filaments over time by solving a linear assignment problem using the information of spatial location of each individual filament. 

Importantly, GraFT can be used to determine and compare properties of filamentous structures in time-resolved image-data across different cell types as well as other filament-based systems; sparse or dense. Therefore, GraFT offers a substantial step towards an automated framework facilitating robust spatiotemporal studies of the actin cytoskeleton in cells. 

# Workflow/Documentation

## Dependencies
The method was written in Python 3.8 and is dependent on the libraries:
- NetworkX
- Scikit-Image
- Scipy
- Pandas
- Numpy
- Matplotlib

I recommend creating a new python environment using conda
```
conda create -n name_env python
```
And install the dependencies
```
conda install NetworkX
conda install numpy
conda install pandas
conda install scipy
```

## Getting started
The script contains a helper script, utils and a main script calling the helper script. Please remember to change the line with the location of the utils script, to the one on your computer.
When importing a dataset, change the line to the location of your data, and import it.
Run the function, and a folder named after your choice will be created in the directory location you defined.
The folders contain the created figures and csv file of the outputs.

# References
refeer to paper here

# Licence
This project is licensed under the terms of the MIT license.
