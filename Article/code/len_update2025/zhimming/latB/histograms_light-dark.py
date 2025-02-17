#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:50:49 2024

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots
import matplotlib as mpl
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

pixelS = 0.065
cmap=sns.color_palette("Set2")

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/zhimming/latB/figs/'

###############################################################################
#
# import files
#
###############################################################################

df_DC = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_control.csv')
df_DC['cell name'] = 'Control'
df_D10 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_10.csv')
df_D10['cell name'] = '0-10 mins'
df_D20 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_20.csv')
df_D20['cell name'] = '10-20 mins'
df_D30 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_30.csv')
df_D30['cell name'] = '20-30 mins'

df_allLight = pd.concat([df_DC,df_D10,df_D20,df_D30], axis=0,ignore_index=True) 


df_DC = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_control.csv')
df_DC['cell name'] = 'Control'
df_D10 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_10.csv')
df_D10['cell name'] = '0-10 mins'
df_D20 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_20.csv')
df_D20['cell name'] = '10-20 mins'
df_D30 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_30.csv')
df_D30['cell name'] = '20-30 mins'

df_allDark = pd.concat([df_DC,df_D10,df_D20,df_D30], axis=0,ignore_index=True) 


###############################################################################
#
# histogram of length of filaments
#
###############################################################################

plt.close('all')

fig, axd = plt.subplot_mosaic("ABCD;EFGH", figsize=(8.27,5))

axd['A'].hist(df_allLight["mean length per filament"][df_allLight['cell name']=='Control']*pixelS,bins=120, density=False,alpha=1,color='#FF6700',label='Control')
axd['B'].hist(df_allLight["mean length per filament"][df_allLight['cell name']=='0-10 mins']*pixelS,bins=120, density=False,alpha=1,color='darkturquoise',label='0-10 mins')
axd['C'].hist(df_allLight["mean length per filament"][df_allLight['cell name']=='10-20 mins']*pixelS,bins=120, density=False,alpha=1,color='indigo',label='10-20 mins')
axd['D'].hist(df_allLight["mean length per filament"][df_allLight['cell name']=='20-30 mins']*pixelS,bins=120, density=False,alpha=1,color='blue',label='20-30 mins')

axd['E'].hist(df_allDark["mean length per filament"][df_allDark['cell name']=='Control']*pixelS,bins=120, density=False,alpha=1,color='#FF6700',label='Control')
axd['F'].hist(df_allDark["mean length per filament"][df_allDark['cell name']=='0-10 mins']*pixelS,bins=120, density=False,alpha=1,color='darkturquoise',label='0-10 mins')
axd['G'].hist(df_allDark["mean length per filament"][df_allDark['cell name']=='10-20 mins']*pixelS,bins=120, density=False,alpha=1,color='indigo',label='10-20 mins')
axd['H'].hist(df_allDark["mean length per filament"][df_allDark['cell name']=='20-30 mins']*pixelS,bins=120, density=False,alpha=1,color='blue',label='20-30 mins')

for nk in ['A','B','C','D','E','F','G','H']:
    threshold_list = [3.3, 6.5, 19.5]
    for s in threshold_list:
        axd[nk].axvline(x=s, color='red', linestyle='--', linewidth=.5, label=f'Threshold {s}')
        
axd['E'].set_xlabel('Length [$\mu$m]')
axd['F'].set_xlabel('Length [$\mu$m]')
axd['G'].set_xlabel('Length [$\mu$m]')
axd['H'].set_xlabel('Length [$\mu$m]')

axd['A'].set_ylabel("Counts, Control Light")
axd['B'].set_ylabel("Counts, 0-10 mins Light")
axd['C'].set_ylabel("Counts, 10-20 mins Light")
axd['D'].set_ylabel("Counts, 20-30 mins Light")

axd['E'].set_ylabel("Counts, Control Dark")
axd['F'].set_ylabel("Counts, 0-10 mins Dark")
axd['G'].set_ylabel("Counts, 10-20 mins Dark")
axd['H'].set_ylabel("Counts, 20-30 mins Dark")
#axd['A'].sharex(axd['B'])
#axd['B'].sharex(axd['C'])

for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(pathsave + 'histograms_Latb.pdf')

