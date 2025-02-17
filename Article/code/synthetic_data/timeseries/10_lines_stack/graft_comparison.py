#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:50:38 2024

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

path="wheredatautilsis"
sys.path.append(path)

import dataUtils

plt.close('all')


box_size = 500


###############################################################################
#
# identification for 5 lines data
#
###############################################################################



files_csv = overpath+'/noise001/timeseries/10_lines_stack/'

for m in range(10):
    savepathLines = overpath+'/noise001/timeseries/10_lines_stack/GraFT/10_lines_0{0}/'.format(m)
    imgs_list = ['{0}_10lines_5movement.csv.tiff'.format(m)]

    csv_list = ['{0}_10lines_5movement.csv_line_position.csv'.format(m)]

    graph_path = overpath+'/noise001/timeseries/10_lines_stack/GraFT/10_lines_0{0}/'.format(m)
    graph_list = ['tagged_graph.gpickle']

    pos_list = ['posL.gpickle']

    dataUtils.identification_LAP(csv_list=csv_list,files_csv=files_csv,savepathLines=savepathLines,graph_path=graph_path,graph_list=graph_list,pos_list=pos_list,box_size=500)

