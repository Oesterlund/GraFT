#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:00:24 2024

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
overpath = 'define_this_path'
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


figsSave = overpath+'/others_code/DeFiNe/noise001/figs_comparisons/'

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


pathS = overpath+'/others_code/DeFiNe/noise001/comparison/files/'
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
    # remove the ones that does not have a match from the snakes definition side(these are lines from true image not belonging to a snake)
    dflInesICov = dflInesI[~np.isnan(dflInesI['match index'])].copy()
 
    
    #dflInesICov['FS_coverage'] = dflInesICov['FS_coverage'].fillna(0)
    dflInesICov.loc[:, 'FS_coverage'] = dflInesICov['FS_coverage'].fillna(0)
    
    overlapList[m] = np.median(dflInesICov['overlap ratio'])
    nooverlapList[m] = len(dflInesICov['overlap ratio'][(dflInesICov['overlap ratio'] == 0)]) / len(np.unique(dflInesICov['true index'][pd.notna(dflInesICov['true index'])])) 

    covL[m] = np.median(dflInesICov['FS_coverage'])
    
    #match2 = re.search(pattern2, selected_values['line type'][m])
    selLin[m] = selected_values['line type'][m] #match2.group(0)
    
    #match3 = re.search(pattern, df_JIT['name'][m])
    #lineLJI[m] = int(match3.group(1))
    #frameLJI[m] = int(match3.group(2))
    
nooverlapList[nooverlapList >1]=1
df_lines = pd.DataFrame({
    'name': namesL,
    'Line': lineL,
    'frame': frameL,
    'FS coverage DeFiNe': covL,
    'FS Overlap coverage DeFiNe': overlapList,
    'DeFiNe ratio no overlap': nooverlapList
})

selected_values['Line'] = selected_values['line type'] 
selected_values.columns

df_DeFiNe = pd.merge(df_lines, selected_values, on=['frame', 'Line'], how='inner')

df_DeFiNe.to_csv(overpath+'/others_code/DeFiNe/noise001/data_DeFiNe.csv')

df_TSOAX = pd.read_csv(overpath+'/others_code/TSOAX/data_TSOAX_graft.csv')

dfTG = df_DeFiNe.merge(df_TSOAX, how='left', on=([ 'frame', 'Line type','grouping','density groups image','density full image']))

'''
df_graft.columns
df_DeFiNe.columns
dfTG = df_DeFiNe.merge(df_graft, how='left', on=([ 'frame', 'Line type','grouping','density groups image','density full image']))

df_DeFiNe.columns.values
dfTG.columns.values
df_graft.columns.values
'''
dfTG = dfTG.sort_values(by=['density groups image'])


plt.figure(figsize=(8.27,5))
plt.scatter(dfTG['grouping'],dfTG['FS coverage DeFiNe'], s=50,alpha=0.8,label='TSOAX')
plt.scatter(dfTG['grouping'],dfTG['FS coverage'], s=50,alpha=0.8,label='GraFT')
plt.legend()



fig, axs = plt.subplot_mosaic("BC;DE",  figsize=(8.27,6))

sns.lineplot(data=dfTG, x='grouping', y='FS coverage',estimator='mean', errorbar=('ci', 90),label='GraFT',ax=axs['B']) #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='FS coverage DeFiNe',estimator='mean', errorbar=('ci', 90),label='DeFiNe',ax=axs['B']) #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='FS coverage TSOAX',estimator='mean', errorbar=('ci', 90),label='TSOAX',ax=axs['B']) #for 95% confidence interval

axs['B'].tick_params(axis='x', labelrotation=30)
axs['B'].legend(frameon=False)
#plt.tight_layout()
#plt.savefig(figsSave+'FS_errorplot_density_DeFiNe.png')


sns.lineplot(data=dfTG, x='grouping', y='Overlap coverage',estimator='mean', errorbar=('ci', 90),label='GraFT',ax=axs['C']) #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='FS Overlap coverage DeFiNe',estimator='mean', errorbar=('ci', 90),label='DeFiNe',ax=axs['C']) #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='FS Overlap coverage',estimator='mean', errorbar=('ci', 90),label='TSOAX',ax=axs['C']) #for 95% confidence interval

axs['C'].tick_params(axis='x', labelrotation=30)
axs['C'].legend().remove()
#plt.tight_layout()
#plt.savefig(figsSave+'FS_overlap_errorplot_density_DeFiNe.png')

sns.lineplot(data=dfTG, x='grouping', y='ratio no overlap',estimator='mean', errorbar=('ci', 90),label='GraFT',ax=axs['D'])
sns.lineplot(data=dfTG, x='grouping', y='DeFiNe ratio no overlap',estimator='mean', errorbar=('ci', 90),label='DeFiNe',ax=axs['D'])
sns.lineplot(data=dfTG, x='grouping', y='TSOAX ratio no overlap',estimator='mean', errorbar=('ci', 90),label='TSOAX',ax=axs['D'])

axs['D'].tick_params(axis='x', labelrotation=30)
axs['D'].legend().remove()
#plt.tight_layout()
#plt.savefig(figsSave+'FS_no_overlap_ratio_errorplot_density_DeFiNe.png')

sns.lineplot(data=dfTG, x='grouping', y='Jaccard Index',estimator='mean', errorbar=('ci', 90),label='GraFT',ax=axs['E']) #for 95% confidence interval
sns.lineplot(data=dfTG, x='grouping', y='line_intensity',estimator='mean',c=((0.8705882352941177, 0.5607843137254902, 0.0196078431372549)), errorbar=('ci', 90),label='TSOAX',ax=axs['E']) #for 95% confidence interval
plt.ylim(0.65,1)

axs['E'].tick_params(axis='x', labelrotation=30)
axs['E'].legend(frameon=False)

plt.savefig(figsSave+'together.png')

axs['C'].sharey(axs['B'])
axs['D'].sharey(axs['B'])

axs['B'].set_xlabel(None)
axs['C'].set_xlabel(None)
axs['D'].set_xlabel("Density image")
axs['E'].set_xlabel("Density image")

axs['B'].set_ylabel("Filament matched coverage")
axs['C'].set_ylabel("Filament coverage")
axs['D'].set_ylabel("No coverage to true filament")
axs['E'].set_ylabel("Jaccard Index")


axs['B'].tick_params(labelbottom=False)
axs['C'].tick_params(labelbottom=False)



for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')
    
plt.tight_layout()

plt.savefig(figsSave+'together.png')

'''
from scipy.stats import mannwhitneyu, levene, shapiro, ttest_ind

for m in df_TSOAX['grouping'].unique():
    
    df_now = df_TSOAX[df_TSOAX['grouping']==m]
    
    #u_statistic, p_value = mannwhitneyu(df_now['FS coverage TSOAX'], df_now['FS coverage'])
    u_statistic, p_value = ttest_ind(df_now['FS coverage TSOAX'], df_now['FS coverage'])
    print(m, u_statistic, p_value)
    if(p_value<=0.1):
        print('significant p-val')
        
        
for m in df_TSOAX['grouping'].unique():
    
    df_now = df_TSOAX[df_TSOAX['grouping']==m]
    
    u_statistic, p_value = mannwhitneyu(df_now['FS coverage TSOAX'], df_now['FS coverage'])
    #u_statistic, p_value = levene(df_now['FS coverage TSOAX'], df_now['FS coverage'])
    print(m, u_statistic, p_value)
    if(p_value<=0.1):
        print('significant p-val')
    
    
for m in df_TSOAX['grouping'].unique():
     
    df_now = df_TSOAX[df_TSOAX['grouping']==m]
    
        
    shapiro_statistic1, shapiro_p_value1 = shapiro(df_now['FS coverage TSOAX'])
    shapiro_statistic2, shapiro_p_value2 = shapiro(df_now['FS coverage'])
    print(f"Shapiro-Wilk test for TSOAX: statistic: {shapiro_statistic1}, p-value: {shapiro_p_value1}")
    print(f"Shapiro-Wilk test for GraFT: statistic: {shapiro_statistic2}, p-value: {shapiro_p_value2}")
'''