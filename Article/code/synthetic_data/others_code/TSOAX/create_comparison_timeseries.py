#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:13:00 2024

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

TSOAX_data.snakes_time(path_snake=overpath+'/others_code/TSOAX/timeseries/test/timeseries_test-snakes.txt',
            savepath = overpath+'/others_code/TSOAX/timeseries/test/tracked/')