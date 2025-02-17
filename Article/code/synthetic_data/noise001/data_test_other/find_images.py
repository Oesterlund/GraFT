#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:37:37 2024

@author: isabella
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.stats import mannwhitneyu
#from patsy import dmatrices
#from statannotations.Annotator import Annotator
import astropy.stats
import scienceplots
from scipy import stats
import pickle
import tifffile

plt.close('all')

plt.style.use(['science','nature']) # sans-serif font
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


cmap=sns.color_palette("colorblind")

###############################################################################
#
# choose images
#
###############################################################################
'''
df_FSfull = pd.read_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/pooled/density_pooled.csv') 


df_FSfull.columns.tolist()

np.unique(df_FSfull['grouping'])
UnG = np.unique(df_FSfull['density groups image'])

###################
# choose 10 from each group, spread out over the density
selected_values = pd.DataFrame()
for m in UnG:
    
    df_0 = df_FSfull[df_FSfull['density groups image']==m]
    
    
    df_0['Density full image']
    
    median_value = df_0['Density full image'].median()
    std_value = df_0['Density full image'].std()
    
    
    # Select 3 values around the median (within 1 standard deviation)
    values_around_median = df_0[(df_0['Density full image'] >= median_value - 0.5*std_value) & (df_0['Density full image'] <= median_value + 0.5*std_value)].sample(10)
    
    print(median_value,values_around_median[['line type','frame','Density full image']])
    
    selected_values = pd.concat([values_around_median, selected_values]).reset_index(drop=True)

selected_values.to_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/data_test_other/' + 'which_frames_chosen.csv', index=False)  
'''
################
# for define

selected_values = pd.read_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/data_test_other/which_frames_chosen.csv')

graph_path = '/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/'

graph_define = np.zeros(len(selected_values)).astype(object)
pos_define = np.zeros(len(selected_values)).astype(object)
for l in range(len(selected_values)):
    curDF = selected_values.iloc[l]
    
    frameVal = int(curDF['frame'])
    
    if(curDF['line type']==5):
        pathAdd = 'noise5'
        
    if(curDF['line type']==10):
        pathAdd = 'noise10'
        
    if(curDF['line type']==20):
        pathAdd = 'noise20'
        
    elif(curDF['line type']==30):
        pathAdd = 'noise30'
        
    elif(curDF['line type']==40):
        pathAdd = 'noise40'
    
    pathCur = graph_path+pathAdd+'/graph/'
    
    if(curDF['frame']<101):
        val = 0
    elif(101<=curDF['frame']<=200):
        val = 1
        frameVal = frameVal - 100
    elif(201<=curDF['frame']<=300):
        val = 2
        frameVal = frameVal - 200
    elif(301<=curDF['frame']<=400):
        val = 3
        frameVal = frameVal - 300
    elif(401<=curDF['frame']<=500):
        val = 4
        frameVal = frameVal - 400
        
    allgraphTags = pd.read_pickle(pathCur+'{0}_graphTagg.gpickle'.format(val))
    allpos = pd.read_pickle(pathCur+'{0}_posL.gpickle'.format(val))
    graph_define[l] = allgraphTags[frameVal-1]
    pos_define[l] = allpos[frameVal-1]

pickle.dump(graph_define, open('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/data_test_other/'+'graph_for_DeFiNe_001.gpickle', 'wb'))
pickle.dump(pos_define, open('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/data_test_other/'+'pos_for_DeFiNe_001.gpickle', 'wb'))

###############################################################################
#
# create tiff format images
#
###############################################################################

img_path = '/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/'

selected_values = pd.read_csv('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/data_test_other/which_frames_chosen.csv')

savepathImg = '/home/isabella/Documents/PLEN/dfs/others_code/TSOAX/'

for m in np.unique(selected_values['line type']):
    
    selected_valuesCur = selected_values[selected_values['line type']==m]
    
    imgCur = np.zeros((len(selected_valuesCur),500,500))
    imgCurNo = np.zeros((len(selected_valuesCur),500,500))
    for l in range(len(selected_valuesCur)):
        curDF = selected_valuesCur.iloc[l]
        
        frameVal = int(curDF['frame'])
        
        nameFrame = '_' + str(frameVal)
        
        if(curDF['line type']==5):
            pathAdd = 'noise5/'
            filename = 'line_5'
            
        if(curDF['line type']==10):
            pathAdd = 'noise10/'
            filename = 'line_10'
            
        if(curDF['line type']==20):
            pathAdd = 'noise20/'
            filename = 'line_20'
            
        elif(curDF['line type']==30):
            pathAdd = 'noise30/'
            filename = 'line_30'
            
        elif(curDF['line type']==40):
            pathAdd = 'noise40/'
            filename = 'line_40'
    
        
        if(curDF['frame']<101):
            nameAdd = '100_lines_intermediate.tiff'
            nameNo = '100_lines_intermediate_nonoise.tiff'
            val = 0
        elif(101<=curDF['frame']<=200):
            nameAdd = '200_lines_intermediate.tiff'
            nameNo = '200_lines_intermediate_nonoise.tiff'
            val = 1
            frameVal = frameVal - 100
        elif(201<=curDF['frame']<=300):
            nameAdd = '300_lines_intermediate.tiff'
            nameNo = '300_lines_intermediate_nonoise.tiff'
            val = 2
            frameVal = frameVal - 200
        elif(301<=curDF['frame']<=400):
            nameAdd = '400_lines_intermediate.tiff'
            nameNo = '400_lines_intermediate_nonoise.tiff'
            val = 3
            frameVal = frameVal - 300
        elif(401<=curDF['frame']<=500):
            nameAdd = '500_lines_intermediate.tiff'
            nameNo = '500_lines_intermediate_nonoise.tiff'
            val = 4
            frameVal = frameVal - 400
        
        pathCur = img_path + pathAdd +nameAdd
        
        with tifffile.TiffFile(pathCur) as tif:
            img_o = tif.asarray()
    
        imgCur[l] = img_o[frameVal-1]
        
        with tifffile.TiffFile(img_path + pathAdd +nameNo) as tif:
            img_nonoise = tif.asarray()
        imgCurNo[l] = img_nonoise[frameVal-1]
        
        #save individual frame
        tifffile.imwrite(savepathImg + 'single/' + filename + nameFrame + '.tiff', imgCur[l].astype(np.uint8), photometric='minisblack',imagej=True)
        # save individual frame nonoise
        tifffile.imwrite(savepathImg + 'nonoise/' + filename + nameFrame + '.tiff', imgCurNo[l].astype(np.uint8), photometric='minisblack',imagej=True)

    #imgCur = imgCur.astype(np.uint8)
    
    # save as bundle
    #tifffile.imwrite(savepathImg+filename + '.tiff', imgCur, photometric='minisblack',imagej=True)
        
    