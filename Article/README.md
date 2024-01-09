# Article information of data & scripts

## Data
The data is shared through Zenodo, with DOI: [10.5281/zenodo.10476058](https://doi.org/10.5281/zenodo.10476058)

On Zenodo there are 5 zipped folders, one for the Latrunculin B data, LatB, it contains two subfolders dark and light. And then three folders for virulence factor (VF) treatment; DMSO, DSF and flg22. The last folder is seedling position data, seedling_pos.

The time-series dataset of 5-day old seedlings of Arabidopsis Thaliana with the green marker line mNeonGreen-FABD2 on actin-cytoskeleton. The seedlings are imaged at three different locations along the stem, upper (young cells), midlle (expanding cells) and at the bottom (mature cells) close to the root start. The seedlings wher imaged using an inverted 3i Marianas Spinning Disk confocal microscope using a 60X objective. The resolution is 0.217 micrometers with 1 sec. intervals.

The time-series dataset LatB is of 5-day old seedlings of Arabidopsis Thaliana with Lifeact-Venus labelled actin-cytoskeleton. They where etiolated and light-grown. 

The time-series dataset for VF treatment is of 5-day old  seedlings (etiolated and light-grown) with Lifeact-Venus labelled actin-cytoskeleton. The seedlings where treated with DSF, flg22 and DMSO for control.

Both the LatB and VF treated seedlings where imaged using Nikon ECLIPSE Ti-2 inverted microscope with a resolution of 0.065 micrometers with 2 sec. intervals.

## Scripts
In the folder code/ all the scripts for data generating and later processing of the data can be found. There are two main folders, depending on the experiment. Mariana/ is the folder containing scripts for seedling position data and Nikon contains the LatB and VF data.
The run*.py is the initial scripts running GraFT. This script take the image-data and generate data that are then later processed in the rest of the scripts. The script timeseries*.py creates a dataset cleaned and pooled together for easy access in the later scripts. The rest of the scripts are for figure creation.
