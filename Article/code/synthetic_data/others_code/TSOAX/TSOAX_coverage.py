#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:06:12 2024

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
overpath = 'define_this_path'
figsSave = overpath+'/others_code/TSOAX/figs_comparisons/'

###############################################################################
#
# 
#
###############################################################################
selected_values = pd.read_csv(overpath+'/noise001/data_test_other/which_frames_chosen_update.csv')
# check for nan
selected_values[np.isnan(selected_values['density groups image'])]
selected_values.columns

df_graft = pd.read_csv(overpath+'/noise001/pooled/' + 'pooled_data.csv')

df_JIT = pd.read_csv(overpath+'/others_code/TSOAX/TSOAX_JI.csv')

pathS = overpath+'/others_code/TSOAX/density/'
namesL = os.listdir(pathS)
pattern = r'(\d+)_(\d+)'
pattern2 = r'(\d+)'
covL = np.zeros(len(namesL))
lineL = np.zeros(len(namesL))
frameL = np.zeros(len(namesL))
selLin = np.zeros(len(namesL))
lineLJI = np.zeros(len(namesL))
frameLJI = np.zeros(len(namesL))
overlapList = np.zeros(len(namesL))
nooverlapList = np.zeros(len(namesL))
for m in range(len(namesL)):
    match = re.search(pattern, namesL[m])
    lineL[m] = int(match.group(1))
    frameL[m] = int(match.group(2))
    dflInesI = pd.read_csv(pathS + namesL[m])
    # remove the ones that idoes not have a match from the snakes definition side(these are lines from true image not belonging to a snake)
    dflInesICov = dflInesI[~np.isnan(dflInesI['best_match_line1_id'])]

    dflInesICov['FS_coverage'] = dflInesICov['overlap_count']#/dflInesICov['true_len']
    
    dflInesICov['FS_coverage'] = dflInesICov['FS_coverage'].fillna(0)
    
    overlapList[m] = np.median(dflInesICov['overlap ratio'])
    nooverlapList[m] = len(dflInesICov['overlap ratio'][(dflInesICov['FS_coverage'] == 0) & (dflInesICov['overlap ratio']<=0.1) ]) / len(np.unique(dflInesICov['line2_id'][pd.notna(dflInesICov['line2_id'])])) 
    

    covL[m] = np.median(dflInesICov['FS_coverage'])
    
    #match2 = re.search(pattern2, selected_values['line type'][m])
    selLin[m] = selected_values['line type'][m] #match2.group(0)
    
    match3 = re.search(pattern, df_JIT['name'][m])
    lineLJI[m] = int(match3.group(1))
    frameLJI[m] = int(match3.group(2))
    
df_lines = pd.DataFrame({
    'name': namesL,
    'Line': lineL,
    'frame': frameL,
    'FS coverage TSOAX': covL,
    'FS Overlap coverage': overlapList,
    'TSOAX ratio no overlap': nooverlapList
})

selected_values['Line'] = selected_values['line type'] 
df_JIT['Line'] = df_JIT['Line type'] 
df_JIT['frame'] = df_JIT['frame no'] 

df_JIT = df_JIT.drop(['Line type'],axis=1)

# update index, according to error
#df_lines['frame'] = df_lines['frame'].apply(lambda x: x-1 if 0 < x < 101 else x)
df_JIT['frame'] = df_JIT['frame'].apply(lambda x: x-1 if 0 < x < 101 else x)

df_TSOAX = pd.merge(df_lines, selected_values, on=['frame', 'Line'], how='inner')
df_TSOAX = pd.merge(df_TSOAX, df_JIT, on=['frame', 'Line'], how='inner')

df_graft.columns
df_TSOAX.columns
dfTG = df_TSOAX.merge(df_graft, how='left', on=([ 'frame', 'Line type','grouping','density groups image','density full image']))

dfTG.to_csv(overpath+'/others_code/TSOAX/data_TSOAX_graft.csv', index=False)  

df_TSOAX.columns.values
dfTG.columns.values
df_graft.columns.values
dfTG = dfTG.sort_values(by=['density groups image'])


plt.figure(figsize=(8.27,5))
plt.scatter(dfTG['grouping'],dfTG['FS coverage TSOAX'], s=50,alpha=0.8,label='TSOAX')
plt.scatter(dfTG['grouping'],dfTG['FS coverage'], s=50,alpha=0.8,label='GraFT')
plt.legend()

plt.figure(figsize=(10,5))
sns.lineplot(data=dfTG, x='grouping', y='FS coverage TSOAX',estimator='mean', errorbar=('ci', 90),n_boot=5000, label='TSOAX') #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='FS coverage',estimator='mean', errorbar=('ci', 90), n_boot=5000, label='GraFT') #for 95% confidence interval
plt.legend()
plt.xlabel('density image')
plt.ylabel('filament coverage')
plt.savefig(figsSave+'FS_errorplot_density_TSOAX.png')


plt.figure(figsize=(10,5))
sns.lineplot(data=dfTG, x='grouping', y='line_intensity',estimator='mean', errorbar=('ci', 90),label='TSOAX') #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='Jaccard Index',estimator='mean', errorbar=('ci', 90),label='GraFT') #for 95% confidence interval

plt.xlabel('density image')
plt.ylabel('Jaccard Index')
plt.legend()
plt.savefig(figsSave+'JI_errorplot_density_TSOAX.png')


plt.figure(figsize=(10,5))
sns.lineplot(data=dfTG, x='grouping', y='FS Overlap coverage',estimator='mean', errorbar=('ci', 90),label='TSOAX') #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='Overlap coverage',estimator='mean', errorbar=('ci', 90),label='GraFT') #for 95% confidence interval

plt.xlabel('density image')
plt.ylabel('Filament coverage overlap')
plt.legend()
plt.savefig(figsSave+'FS_overlap_errorplot_density_TSOAX.png')


plt.figure(figsize=(10,5))
sns.lineplot(data=dfTG, x='grouping', y='TSOAX ratio no overlap',estimator='mean', errorbar=('ci', 90),label='TSOAX')
sns.lineplot(data=dfTG, x='grouping', y='ratio no overlap',estimator='mean', errorbar=('ci', 90),label='GraFT')
plt.xlabel('density image')
plt.ylabel('ratio no overlap to true FS')
plt.tight_layout()
plt.legend()
plt.savefig(figsSave+'FS_no_overlap_ratio_errorplot_density_TSOAX.png')
