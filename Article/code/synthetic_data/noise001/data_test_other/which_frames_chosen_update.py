#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:44:01 2024

@author: isabella
"""
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.colors as colors
#import skimage.io as io
import scienceplots
import tifffile
import seaborn as sns
import os
import re

plt.close('all')

figsSave = '/home/isabella/Documents/PLEN/dfs/others_code/imgs/figs_comparisons/'

###############################################################################
#
# 
#
###############################################################################

selected_values = pd.read_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/data_test_other/which_frames_chosen.csv')
#selected_values = selected_values.drop(['FS coverage'], axis=1)
selected_values.columns

selected_values = selected_values.drop(['density groups image', 'Density full image','grouping'], axis=1)

selected_values[selected_values['frame']==0]
# move the frame value by one for values under 100, as they are indexed wrongly.
selected_values['frame'] = selected_values['frame'].apply(lambda x: x-1 if 0 < x < 101 else x)

df_FSfull = pd.read_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/pooled/pooled_data.csv') 
df_FSfull.columns

df_FSfull['line type'] = 0

pattern = r'(\d+)'
lineN = np.zeros(len(df_FSfull))
for m in range(len(df_FSfull)):
    match = re.search(pattern, df_FSfull['Line type'][m])
    lineN[m] = match.group(0)
    
df_FSfull['line type'] = lineN

selected_values_new = selected_values.merge(df_FSfull[[ 'frame', 'line type','Line type', 'grouping','density groups image','density full image']], how='left', on=([ 'frame', 'line type']))

density_counts = selected_values_new['density groups image'].value_counts().sort_index()


selected_values_new.to_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/data_test_other/' + 'which_frames_chosen_update.csv', index=False)  
