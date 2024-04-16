#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 08:57:59 2023

@author: pgf840
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io

from graft import utilsF
from graft.main import create_all, create_all_still

# Get the directory containing this script.
base_path = os.path.dirname(os.path.abspath(__file__))

plt.close('all')

SIGMA = 1.0  # tubeness filter width
SMALL = 50.0  # cluster removal  


if __name__ == '__main__':
    ###############################################################################
    #
    # load in and run functions
    #
    ###############################################################################

    ######################
    # timeseries

    img_o = io.imread(os.path.join(base_path, "tiff", "timeseries.tif"))
    maskDraw = np.ones((img_o.shape[1:3]))
    create_all(pathsave=os.path.join(base_path, "timeseries"),
               img_o=img_o,
               maskDraw=maskDraw,
               size=6,eps=200,thresh_top=0.5,sigma=SIGMA,small=SMALL,angleA=140,overlap=4,max_cost=100,
               name_cell='in silico time')

    ######################
    # one image

    img = io.imread(os.path.join(base_path, "tiff", "timeseries.tif"))
    img_still = img_o[0]
    maskDraw = np.ones((img.shape[1:3]))
    create_all_still(pathsave=os.path.join(base_path, "still"),
               img_o=img_still,
               maskDraw=maskDraw,
               size=6,eps=200,thresh_top=0.5,sigma=SIGMA,small=SMALL,angleA=140,overlap=4,
               name_cell='in silico still')
