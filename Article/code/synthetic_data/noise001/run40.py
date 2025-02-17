#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: isabella
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

path="where_dataUtils_is"
sys.path.append(path)

import dataUtils

overpath = 'define_this_path'

plt.close('all')

box_size = 500

###############################################################################
#
# use GraFT to create graphs
#
###############################################################################

# load in test data
graphSave = overpath+'/noise001/noise40/graph/'

path_imgs = overpath+'/noise001/noise40/'

imgs_list = ['100_lines_intermediate.tiff','200_lines_intermediate.tiff','300_lines_intermediate.tiff','400_lines_intermediate.tiff','500_lines_intermediate.tiff']

csv_list = ['100_lines_intermediate_line_position.csv','200_lines_intermediate_line_position.csv','300_lines_intermediate_line_position.csv',
              '400_lines_intermediate_line_position.csv','500_lines_intermediate_line_position.csv']

#create tiff files
dataUtils.create_tiffimgs(imgs_list = imgs_list,path_imgs = path_imgs)

dataUtils.use_GraFT(imgs_list=imgs_list,path_imgs=path_imgs,box_size=500,graphSave= graphSave
                    ,size=4,eps=100,thresh_top=0.3,sigma=1.5,small=20)

###############################################################################
#
# identification for 40 lines data
#
###############################################################################

savepathLines = overpath+'/noise40/line_comparisons/'

files_csv = overpath+'/noise001/noise40/'

imgs_list = ['100_lines_intermediate.tiff','200_lines_intermediate.tiff','300_lines_intermediate.tiff','400_lines_intermediate.tiff','500_lines_intermediate.tiff']

csv_list = ['100_lines_intermediate_line_position.csv','200_lines_intermediate_line_position.csv','300_lines_intermediate_line_position.csv',
            '400_lines_intermediate_line_position.csv','500_lines_intermediate_line_position.csv']

graph_path = overpath+'/noise001/noise40/graph/'
graph_list = ['0_graphTagg.gpickle','1_graphTagg.gpickle','2_graphTagg.gpickle','3_graphTagg.gpickle','4_graphTagg.gpickle']

pos_list = ['0_posL.gpickle','1_posL.gpickle','2_posL.gpickle','3_posL.gpickle','4_posL.gpickle']

dataUtils.identification_LAP(csv_list=csv_list,files_csv=files_csv,savepathLines=savepathLines,graph_path=graph_path,graph_list=graph_list,pos_list=pos_list,box_size=500)
