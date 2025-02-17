#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:58:30 2024

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
#import astropy.stats
#import plotly.graph_objects as go
#from astropy import units as u
from statannotations.Annotator import Annotator
import scienceplots
from scipy.stats import skew
###############################################################################
#
# create a merged full dataset, to test best division of data
#
###############################################################################


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

pixelSZhimming = 0.065
cmap=sns.color_palette("Set2")

pathsave = '/home/isabella/Documents/PLEN/dfs/data/paper_figs/histograms/'

###############################################################################
#
# import files
#
###############################################################################


###
# load in latb data

df_DC = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_control.csv')
df_DC['cell name'] = 'Control'
df_D10 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_10.csv')
df_D10['cell name'] = '0-10 mins'
df_D20 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_20.csv')
df_D20['cell name'] = '10-20 mins'
df_D30 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_30.csv')
df_D30['cell name'] = '20-30 mins'

df_allLight = pd.concat([df_DC,df_D10,df_D20,df_D30], axis=0,ignore_index=True) 
df_allLight['type'] = 'LatB'

df_DC = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_control.csv')
df_DC['cell name'] = 'Control'
df_D10 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_10.csv')
df_D10['cell name'] = '0-10 mins'
df_D20 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_20.csv')
df_D20['cell name'] = '10-20 mins'
df_D30 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_30.csv')
df_D30['cell name'] = '20-30 mins'

df_allDark = pd.concat([df_DC,df_D10,df_D20,df_D30], axis=0,ignore_index=True) 
df_allDark['type'] = 'LatB'

###
# load in DMSO data

df_con = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/frames_DMSOcontrol.csv')
df_con['cell name'] = 'Control'
df_DSF = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/frames_DSF.csv')
df_DSF['cell name'] = 'DSF'
df_flg22 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/frames_flg22.csv')
df_flg22['cell name'] = 'flg22'


df_allDMSO = pd.concat([df_con,df_DSF,df_flg22], axis=0,ignore_index=True) 
df_allDMSO['type'] = 'DMSO'

df_allInt = pd.concat([df_allLight,df_allDark,df_allDMSO], axis=0,ignore_index=True) 
df_allInt["mean length per filament"] = df_allInt["mean length per filament"] * pixelSZhimming

#####
# this data has a different pixel size
pixelSIsa = 0.217
df_Dtop = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/frames_top.csv')
df_Dtop['cell name'] = 'Young'
df_Dmid = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/frames_mid.csv')
df_Dmid['cell name'] = 'Expanding'
df_Dbot = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/frames_bot.csv')
df_Dbot['cell name'] = 'Mature'

df_allHyp = pd.concat([df_Dtop,df_Dmid,df_Dbot], axis=0,ignore_index=True) 
df_allHyp["mean length per filament"] = df_allHyp["mean length per filament"] * pixelSIsa

###############
# merge all together

df_all = pd.concat([df_allInt,df_allHyp], axis=0,ignore_index=True) 

data = df_all["mean length per filament"]

df_all["decile"] = pd.qcut(df_all["mean length per filament"], q=10, labels=False)

# Plot the histogram with clusters
plt.figure(figsize=(10, 6))
for m in range(10):
    plt.hist(df_all[df_all["decile"] == m]["mean length per filament"], bins=30, alpha=0.6, label=m)

plt.xlabel("Length")
plt.ylabel("Frequency")
plt.title("Length Distribution by Groups")
plt.legend()
plt.show()


plt.figure(figsize=(8.27, 4))
plt.hist(df_all["mean length per filament"], bins=800, alpha=1,density=True)
plt.xlim(0,30)
plt.axvline(x=np.max(df_all[df_all["decile"] == 3]["mean length per filament"]), linestyle="--", color="red", label="4th decile")
plt.axvline(x=np.max(df_all[df_all["decile"] == 6]["mean length per filament"]), linestyle="--", color="red", label="7th decile")
plt.axvline(x=np.max(df_all[df_all["decile"] == 8]["mean length per filament"]), linestyle="--", color="red", label="9th decile")
plt.xlabel("Length [$\mu$m]")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(pathsave+'all_pooled_length_data.pdf')

for k in range(10):
    print(np.max(df_all[df_all["decile"] == k]["mean length per filament"]))
    
plt.close('all')
    
# Calculate statistics
mean = np.mean(data)
median = np.median(data)
mode = pd.Series(data).mode()[0]
data_skewness = skew(data)
max_value = np.max(data)
