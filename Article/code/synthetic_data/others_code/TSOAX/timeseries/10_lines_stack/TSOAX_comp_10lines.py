#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:26:53 2024

@author: isabella
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.colors as colors
#import skimage.io as io
#import scienceplots
import tifffile
#import seaborn as sns
import os
import re
from scipy.optimize import linear_sum_assignment
import sys
overpath = 'define_this_path'
path=overpath+"/others_code/TSOAX/"
sys.path.append(path)

import TSOAX_data

#plt.style.use(['science','nature']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('axes', labelsize=12)

params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
plt.rcParams.update(params)

###############################################################################
#
#
#
###############################################################################

pathS = overpath+'/others_code/TSOAX/timeseries/10_lines_stack/TSOAX_files/'
pathNo = overpath+'/noise001/timeseries/10_lines_stack/'
namesL = os.listdir(pathS)
sorted_names = sorted(namesL, key=lambda x: int(x.split('_')[0]))
for m in range(len(namesL)):
    print(m)
    
    TSOAX_data.snakes_time(path_snake=pathS+sorted_names[m],
                           savepath=overpath+'/others_code/TSOAX/timeseries/10_lines_stack/0{0}_10_lines/tracked/'.format(m),
                           savefile = overpath+'/others_code/TSOAX/timeseries/10_lines_stack/density/',
                           images_path=pathNo + '{0}_10lines_5movement.csv_nonoise.tiff'.format(m),
                           pathNonoise=overpath+'/noise001/timeseries/10_lines_stack/{0}_10lines_5movement.csv_nonoise_line_position.csv'.format(m),
                           track_len=20, 
                           filename = sorted_names[m][0:-4],
                           box_size=500)

    