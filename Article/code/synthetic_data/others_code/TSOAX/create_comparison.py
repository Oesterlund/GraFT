#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:13:11 2024

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

pathS = overpath+'/others_code/TSOAX/snakes/'
namesL = os.listdir(pathS)
pattern = r'_(\d+)_(\d+)'

'''
for i in range(len(namesL)):
    print(i )
    match = re.search(pattern, namesL[i])
    lineT = int(match.group(1))
    frameT = int(match.group(2))
    
    # correct error on which one it is
    frameN= frameT-1 if 0 < frameT <= 101 else frameT
    
    if(frameT<=101):
        nonoiseLinePos=100
    elif(frameT==500):
        nonoiseLinePos = 500
    else:
        nonoiseLinePos = ((100+(frameT-1))//100)*100
               
    snakes(path_snake=pathS+namesL[i],
                      linetype = lineT, frame = frameT, frame_graft = frameN,
                      savepath=overpath+'/others_code/TSOAX/density/',
                      imgpath=overpath+'/others_code/TSOAX/nonoise/line_{0}_{1}.tiff'.format(lineT,frameT),
                      pathNonoise=overpath+'/noise001/noise{0}/{1}_lines_intermediate_nonoise_line_position.csv'.format(lineT,nonoiseLinePos),
                      box_size=500)

'''
JI_list = np.zeros(len(namesL))
for m in range(len(namesL)):
    print(m)
    match = re.search(pattern, namesL[m])
    lineT = int(match.group(1))
    frameT = int(match.group(2))
    
    if(frameT<=101):
        nonoiseLinePos=100
    elif(frameT==500):
        nonoiseLinePos = 500
    else:
        nonoiseLinePos = ((100+frameT)//100)*100
    
    JI_list[m] = TSOAX_data.JI_loop(imgpath=overpath+'/others_code/TSOAX/nonoise/line_{0}_{1}.tiff'.format(lineT,frameT), 
                         path_snake=pathS + namesL[m])

lineL = np.zeros(len(namesL))
frameL = np.zeros(len(namesL))
for m in range(len(namesL)):
    print(m)
    match = re.search(pattern, namesL[m])
    lineL[m] = int(match.group(1))
    frameL[m] = int(match.group(2))
    
df_JI = pd.DataFrame({
    'name': namesL,
    'Line type': lineL,
    'frame no': frameL,
    'line_intensity': JI_list
})

df_JI.sort_values(by=['Line type'])

df_JI.to_csv(overpath+'/others_code/TSOAX/TSOAX_JI.csv', index=False) 

