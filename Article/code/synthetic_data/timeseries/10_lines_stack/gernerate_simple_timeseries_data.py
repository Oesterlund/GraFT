#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:27:28 2024

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from skimage.draw import bezier_curve
from skimage.draw import disk
import tifffile
import re
import networkx as nx
import itertools
import sys
import pickle
from scipy.optimize import linear_sum_assignment

overpath = 'define_this_path'
path=overpath+"/noise001/timeseries/"
sys.path.append(path)

import gernerate_simple_timeseries_data

###############################################################################

for n in range(4,10):
    plt.close('all')
    df_data = gernerate_simple_timeseries_data.generate_filamentous_structure(box_size=500, num_lines=5,frame_no=20, movement=5, curvature_factor=0.9)
    
    imgnoise=gernerate_simple_timeseries_data.generate_images_from_data(df_timeseries=df_data, noiseBlur=1,noiseBack=0.005,box_size=500,
                              path=overpath+'/noise001/timeseries/10_lines_stack/',
                              filename='{0}_10lines_5movement.csv'.format(n))
    
    
    imgno = gernerate_simple_timeseries_data.generate_images_from_data(df_timeseries=df_data,noiseBlur=1,noiseBack=0.005,box_size=500,
                              path=overpath+'/noise001/timeseries/10_lines_stack/',
                              filename='{0}_10lines_5movement.csv'.format(n)+'_nonoise',noise='no')
    
