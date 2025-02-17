#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:11:33 2023

@author: isabella
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import re
#import astropy.stats
import plotly.graph_objects as go
#from astropy import units as u
from statannotations.Annotator import Annotator
import scienceplots
import argparse
import matplotlib.ticker as ticker
import scienceplots
import matplotlib as mpl
plt.style.use(['science','nature']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('axes', labelsize=10)
sizeL=10
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
plt.rcParams.update(params)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False  # Disable LaTeX
mpl.rcParams['axes.unicode_minus'] = True  # Ensure minus sign is rendered correctly


cmap=sns.color_palette("colorblind")

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/zhimming/DMSO/figs/'
pixelS = 0.065

figsize=(8.27/2, 2.5)

###############################################################################
#
# import files
#
###############################################################################

df_all = pd.read_csv(pathsave+'df_all.csv', encoding='utf-8')
df_allFullm = pd.read_csv(pathsave+'df_allFullm.csv', encoding='utf-8')
angleFil = pd.read_csv(pathsave+'df_angles.csv', encoding='utf-8')
assFil = pd.read_csv(pathsave+'df_assFil.csv', encoding='utf-8')

###############################################################################
#
# histogram of length of filaments
#
###############################################################################

plt.close('all')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Control']*pixelS,bins=50, density=False,alpha=0.5,label='Control')
#plt.yscale('log')
#plt.xlim(0,1250)
#plt.ylim(0,0.016)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_control.png')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='DSF']*pixelS,bins=50, density=False,alpha=0.5,label='DSF')
#plt.xlim(0,900)
#plt.ylim(0,0.016)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_DSF.png')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='flg22']*pixelS,bins=50, density=False,alpha=0.5,label='flg22')
#plt.xlim(0,900)
#plt.ylim(0,0.016)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_flg22.png')


fig, axd = plt.subplot_mosaic("ABC", figsize=(8.27,3))

axd['A'].hist(df_all["mean length per filament"][df_all['cell name']=='Control']*pixelS,bins=120, density=False,alpha=1,color='#FF6700',label='Control')
axd['B'].hist(df_all["mean length per filament"][df_all['cell name']=='DSF']*pixelS,bins=120, density=False,alpha=1,color='darkturquoise',label='DSF')
axd['C'].hist(df_all["mean length per filament"][df_all['cell name']=='flg22']*pixelS,bins=120, density=False,alpha=1,color='indigo',label='flg22')

for nk in ['A','B','C']:
    threshold_list = [3.3, 6.5, 19.5]
    for s in threshold_list:
        axd[nk].axvline(x=s, color='red', linestyle='--', linewidth=.5, label=f'Threshold {s}')

axd['A'].set_xlabel('Length [$\mu$m]')
axd['B'].set_xlabel('Length [$\mu$m]')
axd['C'].set_xlabel('Length [$\mu$m]')

axd['A'].set_ylabel("Counts, Control")
axd['B'].set_ylabel("Counts, DSF")
axd['C'].set_ylabel("Counts, flg22")

#axd['A'].sharey(axd['B'])
#axd['B'].sharey(axd['C'])

for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(pathsave + 'histograms_DMSO.pdf')

###############################################################################
#
# plots in 4 groups
#
###############################################################################

x = "cell name 3"
hue = "cell name"
hue_order = ['Control', 'DSF','flg22']
order = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

pairs  = [
    [('<3.22', 'Control'), ('<3.22', 'DSF')],
     [('<3.22', 'Control'), ('<3.22', 'flg22')],
     [('<3.22', 'DSF'), ('<3.22', 'flg22')],
     
     [('3.22-6.15', 'Control'), ('3.22-6.15', 'DSF')],
      [('3.22-6.15', 'Control'), ('3.22-6.15', 'flg22')],
      [('3.22-6.15', 'DSF'), ('3.22-6.15', 'flg22')],

     [('6.15-16.26', 'Control'), ('6.15-16.26', 'DSF')],
      [('6.15-16.26', 'Control'), ('6.15-16.26', 'flg22')],
      [('6.15-16.26', 'DSF'), ('6.15-16.26', 'flg22')],
     
     [('>16.26', 'Control'), ('>16.26', 'DSF')],
      [('>16.26', 'Control'), ('>16.26', 'flg22')],
      [('>16.26', 'DSF'), ('>16.26', 'flg22')],
     ]

plt.close('all')

list_plots = ['mean filament bendiness','mean filament length','mean filament movement','filament movement per length','mean intensity','mean intensity per length','mean filament angle']
list_ylabel = ['Mean bendiness ratio','Mean length','Mean movement [\u03bcm/s]','Mean movement per length','Mean intensity','Mean intensity per \u03bcm','Mean filament angle']

#ylim=[[-.1,100],[-0.05,1.4],[0,300]]

for i in range(len(list_plots)):  
    y = list_plots[i]
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_allFullm, x=x, y=y, order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "coral"},palette=cmap)
    annot = Annotator(ax, pairs, data=df_allFullm, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test='Mann-Whitney', verbose=2,fontsize=15,hide_non_significant=True)
    annot.apply_test()
    annot.annotate()
    plt.ylabel(list_ylabel[i],size=sizeL)
    plt.xlabel('')
    #plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    
    if(i==2):
        plt.legend(fontsize=20,loc='best',frameon=False)
        #plt.ylim(-2,125)
    else:
        plt.legend().remove()
    ax.set_xlim(xmin=-0.5,xmax=3.5)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups_stat.png'.format(list_plots[i]))
    
for m in range(len(hue_order)):
    print(hue_order[m], np.mean(df_allFullm['mean filament movement'][df_allFullm['cell name']==hue_order[m]]))
    
cell2 = np.unique(df_allFullm['cell name 2'])
for m in range(len(cell2)):
    print(cell2[m], np.mean(df_allFullm['mean filament movement'][df_allFullm['cell name 2']==cell2[m]]))
    

cell3 = np.unique(df_allFullm['cell name'])
for m in range(len(cell3)):
    print(cell3[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name']==cell3[m]]))
    
cell2 = np.unique(df_allFullm['cell name 2'])
for m in range(len(cell2)):
    print(cell2[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name 2']==cell2[m]]))


###################
# plots
#


plt.close('all')
fig, axd = plt.subplot_mosaic("DEF;GHI", figsize=(8.27,5))


sns.boxplot(data=df_allFullm, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['D'])
annot = Annotator(axd['D'], pairs, data=df_allFullm, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['E'])
annot = Annotator(axd['E'], pairs, data=df_allFullm, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['F'])
annot = Annotator(axd['F'], pairs, data=df_allFullm, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()


sns.boxplot(data=assFil, x='cells', y='Cell density',showfliers = False,medianprops={"color": "coral"},palette=cmap,ax=axd['G'])
sns.stripplot(data=assFil, x="cells", y="Cell density",s=3, dodge=True, ax=axd['G'],color='black')

plot=sns.lineplot(
    data=angleFil, 
    x="name", 
    y="angle", 
    hue="cells", 
    marker='o', 
    markersize=8,
    n_boot=10000, 
    ci=95,
    ax=axd['H']
)
plot.legend_.set_title(None)
plot.legend(fontsize=6,frameon=False, loc="best")

plot=sns.lineplot(
    data=angleFil, 
    x="name", 
    y="circular variance", 
    hue="cells", 
    marker='o', 
    markersize=8,
    n_boot=10000, 
    ci=95,
    ax=axd['I']
)

plot.legend_.set_title(None)
plot.legend().remove()

axd['D'].legend(fontsize=6,frameon=False, loc="best")
axd['E'].legend().remove()
axd['F'].legend().remove()
axd['D'].set_xlim(xmin=-0.45,xmax=3.45)
axd['E'].set_xlim(xmin=-0.45,xmax=3.45)
axd['F'].set_xlim(xmin=-0.45,xmax=3.45)

adjust = 15
axd['D'].tick_params(axis='x', labelrotation=adjust)
axd['E'].tick_params(axis='x', labelrotation=adjust)
axd['F'].tick_params(axis='x', labelrotation=adjust)

#axd['G'].tick_params(axis='x', labelrotation=adjust)
axd['H'].tick_params(axis='x', labelrotation=adjust)
axd['I'].tick_params(axis='x', labelrotation=adjust)


axd['D'].set_xlabel('')
axd['E'].set_xlabel("")
axd['F'].set_xlabel("")
axd['G'].set_xlabel('')
axd['H'].set_xlabel('')
axd['I'].set_xlabel('')

axd['D'].set_ylabel(r'Mean movement [$\mu m/s$]',size=10)
axd['E'].set_ylabel('Mean bendiness ratio',size=10)
axd['F'].set_ylabel(r'Mean intensity per $\mu m$',size=10)

axd['G'].set_ylabel('Cell density',fontsize=10)
axd['H'].set_ylabel("Circular mean angle",fontsize=10)
axd['I'].set_ylabel("Circular mean variance",fontsize=10)


for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')
 
plt.tight_layout()

plt.savefig(pathsave+'groups/'+'DMSO_full_fig.pdf')
