#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:45:11 2023

@author: isabella
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle
import seaborn as sns
import astropy.stats
import plotly.graph_objects as go
from astropy import units as u

plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 

cmap=sns.color_palette("Set2")

size=30

plt.close('all')

###############################################################################
#
# functions
#
###############################################################################

def circ_stat(DataSet):

    data = DataSet["mean filament angle"]*u.deg
    weight = DataSet['mean filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))*180./np.pi
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))

    return mean_angle,var_val

###############################################################################
#
#
#
###############################################################################

import sys
path="/home/isabella/Documents/PLEN/dfs/data/time_dark_seedlings/"
sys.path.append(path)

import utilsF

pathsave = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/figs/'


df_allFullmdark = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/figs/df_allFullm.csv')
df_allFullmlight = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/figs/df_allFullm.csv')

np.unique(df_allFullmdark['cell name 2'])

name_cell = ['Control','0-10 mins','10-20 mins','20-30 mins']
interval = [ '<3.3', '3.3-6.5', '6.5-19.5', '>19.5']
angleD = np.zeros(4*4)
AvarD = np.zeros(4*4)
angleL = np.zeros(4*4)
AvarL = np.zeros(4*4)
l=0
for i in range(len(name_cell)):
    for m in range(len(interval)):
        print(i,m)
        angleD[l],AvarD[l] = circ_stat(df_allFullmdark[df_allFullmdark['cell name 2']==name_cell[i]+' {0}'.format(interval[m])])
        angleL[l],AvarL[l] = circ_stat(df_allFullmlight[df_allFullmlight['cell name 2']==name_cell[i]+' {0}'.format(interval[m])])
        l+=1
        
angleDark = angleD.reshape((4, 4))
angleLight = angleL.reshape((4, 4))
AvarDark = AvarD.reshape((4, 4))
AvarLight = AvarL.reshape((4, 4))

name = ['Light','Dark']
namesD = ['50','50-100','100-300','>300']
name_cell = ['Control','0-10 mins','10-20 mins','20-30 mins']

angleFil=pd.DataFrame()
for i in range(len(name_cell)):
    dataang = {'Angle' : angleLight[i],'Circular variance':AvarLight[i],'name': namesD,'Treatment': name_cell[i],'Type':name[0]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)
for i in range(len(name_cell)):
    dataang = {'Angle' : angleDark[i],'Circular variance':AvarDark[i],'name': namesD,'Treatment': name_cell[i],'Type':name[1]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)


###################
# plots
#

plt.figure(figsize=(12,8))
sns.scatterplot(data=angleFil, x='Treatment', y='Angle',hue='Type',marker='o',alpha=0.7,s=200,palette=cmap[0:2])
sns.scatterplot(x='Treatment', y='Angle',data=angleFil[angleFil['Type']=='Light'].groupby('Treatment', as_index=False)['Angle'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='black')
sns.scatterplot(x='Treatment', y='Angle',data=angleFil[angleFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Angle'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='black')
sns.lineplot(x='Treatment', y='Angle',data=angleFil[angleFil['Type']=='Light'].groupby('Treatment', as_index=False)['Angle'].mean(),color=(0.4, 0.7607843137254902, 0.6470588235294118),alpha=0.5)
sns.lineplot(x='Treatment', y='Angle',data=angleFil[angleFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Angle'].mean(),color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),alpha=.5)

plt.legend(fontsize=size)  
plt.ylabel('Angle',fontsize=35)
plt.ylim(24,60)
plt.xlabel('')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_all.png')



plt.figure(figsize=(12,8))
sns.scatterplot(data=angleFil, x='Treatment', y='Circular variance',hue='Type',marker='o',alpha=0.7,s=200,palette=cmap[0:2])
sns.scatterplot(x='Treatment', y='Circular variance',data=angleFil[angleFil['Type']=='Light'].groupby('Treatment', as_index=False)['Circular variance'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='black')
sns.scatterplot(x='Treatment', y='Circular variance',data=angleFil[angleFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Circular variance'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='black')
sns.lineplot(x='Treatment', y='Circular variance',data=angleFil[angleFil['Type']=='Light'].groupby('Treatment', as_index=False)['Circular variance'].mean(),color=(0.4, 0.7607843137254902, 0.6470588235294118),alpha=0.5)
sns.lineplot(x='Treatment', y='Circular variance',data=angleFil[angleFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Circular variance'].mean(),color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),alpha=.5)
plt.legend(fontsize=size).remove()
plt.ylabel('Circular variance',fontsize=35)
plt.ylim(0.01,0.06)
plt.xlabel('')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_var_all.png')
