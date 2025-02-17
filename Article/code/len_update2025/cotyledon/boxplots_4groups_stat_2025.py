#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:40:30 2023

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

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/cotyledon/figs/'

pixelS = 0.217
figsize=(8.27/2, 2.5)

###############################################################################
#
# import files
#
###############################################################################

df_all = pd.read_csv(pathsave+'df_all.csv', encoding='utf-8')
df_allFullm = pd.read_csv(pathsave+'df_allFullm.csv', encoding='utf-8')
angleFil = pd.read_csv(pathsave+'df_angles.csv', encoding='utf-8')

angleFil2 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/figs/df_angles.csv', encoding='utf-8')
assFil = pd.read_csv(pathsave+'df_assFil.csv', encoding='utf-8')
###############################################################################
#
# histogram of length of filaments
#
###############################################################################

plt.close('all')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Young']*pixelS,bins=50, density=False,alpha=0.5,label='Upper')
#plt.xlim(0,400)
#plt.ylim(0,0.023)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_upper.png')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Expanding']*pixelS,bins=50, density=False,alpha=0.5,label='Middle')
#plt.xlim(0,1300)
#plt.ylim(0,0.023)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_middle.png')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Mature']*pixelS,bins=50, density=False,alpha=0.5,label='Bottom')
#plt.xlim(0,1300)
#plt.ylim(0,0.023)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_bottom.png')

fig, axd = plt.subplot_mosaic("ABC", figsize=(8.27,3))

axd['A'].hist(df_all["mean length per filament"][df_all['cell name']=='Young']*pixelS,bins=120, density=False,alpha=1,color='#FF6700',label='Upper')
axd['B'].hist(df_all["mean length per filament"][df_all['cell name']=='Expanding']*pixelS,bins=120, density=False,alpha=1,color='darkturquoise', label='Middle')
axd['C'].hist(df_all["mean length per filament"][df_all['cell name']=='Mature']*pixelS,bins=120, density=False,alpha=1, color='indigo',label='Bottom')

for nk in ['A','B','C']:
    threshold_list = [3.2164406150564564, 6.148265832210858, 16.262615814598842]
    for s in threshold_list:
        axd[nk].axvline(x=s, color='red', linestyle='--', linewidth=.5, label=f'Threshold {s}')

axd['A'].set_xlabel('Length [$\mu$m]')
axd['B'].set_xlabel('Length [$\mu$m]')
axd['C'].set_xlabel('Length [$\mu$m]')

axd['A'].set_ylabel("Counts, Young")
axd['B'].set_ylabel("Counts, Expanding")
axd['C'].set_ylabel("Counts, Mature")
#axd['A'].sharey(axd['B'])
#axd['B'].sharey(axd['C'])

for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(pathsave + 'histograms_cotyledon.pdf')


#####################
# change in angle and length per filament

x = "cell name 3"
hue = "cell name"
hue_order = ['Young', 'Expanding', 'Mature']
order = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

pairs  = [
    [('<3.22', 'Mature'), ('<3.22', 'Expanding')],
     [('<3.22', 'Mature'), ('<3.22', 'Young')],
     [('<3.22', 'Expanding'), ('<3.22', 'Young')],
     [('3.22-6.15', 'Mature'), ('3.22-6.15', 'Expanding')],
     [('3.22-6.15', 'Mature'), ('3.22-6.15', 'Young')],
     [('3.22-6.15', 'Expanding'), ('3.22-6.15', 'Young')],
     [('6.15-16.26', 'Mature'), ('6.15-16.26', 'Expanding')],
     [('6.15-16.26', 'Mature'), ('6.15-16.26', 'Young')],
     [('6.15-16.26', 'Expanding'), ('6.15-16.26', 'Young')],
     [('>16.26', 'Mature'), ('>16.26', 'Expanding')],
     [('>16.26', 'Mature'), ('>16.26', 'Young')],
     [('>16.26', 'Expanding'), ('>16.26', 'Young')],
     ]

list_plots = ['mean angle change per filament','mean length change per filament','median length change per filament','mean length per filament']
list_ylabel = ['Mean angle change','Mean length change','Median length change','Mean filament length']

for i in range(len(list_plots)):  
    y = list_plots[i]
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_all, x=x, y=y, order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "coral"},palette=cmap)
    annot = Annotator(ax, pairs, data=df_all, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test='Mann-Whitney', verbose=2)
    annot.apply_test()
    annot.annotate()
    plt.ylabel(list_ylabel[i],size=12)
    plt.xlabel('')
    #plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups_stat.png'.format(list_ylabel[i]))


###############################################################################
#
# plots in 4 groups
#
###############################################################################

plt.close('all')

x = "cell name 3"
hue = "cell name"
hue_order = ['Young', 'Expanding', 'Mature']
order = ['<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

pairs  = [
    [('<3.22', 'Mature'), ('<3.22', 'Expanding')],
     [('<3.22', 'Mature'), ('<3.22', 'Young')],
     [('<3.22', 'Expanding'), ('<3.22', 'Young')],
     [('3.22-6.15', 'Mature'), ('3.22-6.15', 'Expanding')],
     [('3.22-6.15', 'Mature'), ('3.22-6.15', 'Young')],
     [('3.22-6.15', 'Expanding'), ('3.22-6.15', 'Young')],
     [('6.15-16.26', 'Mature'), ('6.15-16.26', 'Expanding')],
     [('6.15-16.26', 'Mature'), ('6.15-16.26', 'Young')],
     [('6.15-16.26', 'Expanding'), ('6.15-16.26', 'Young')],
     [('>16.26', 'Mature'), ('>16.26', 'Expanding')],
     [('>16.26', 'Mature'), ('>16.26', 'Young')],
     [('>16.26', 'Expanding'), ('>16.26', 'Young')],
     ]


list_plots = ['mean filament bendiness','mean filament length','mean filament movement','filament movement per length','mean intensity','mean intensity per length','mean filament angle']
list_ylabel = ['Mean bendiness ratio','Mean length',r'Mean movement [$\mu m/s$]','Mean movement per length','Mean intensity',r'Mean intensity per $\mu m$','Mean filament angle']

for i in range(len(list_plots)):  
    y = list_plots[i]
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_allFullm, x=x, y=y, order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "coral"},palette=cmap)
    annot = Annotator(ax, pairs, data=df_allFullm, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test='Mann-Whitney', verbose=2,fontsize=15,hide_non_significant=True)
    annot.apply_test()
    annot.annotate()
    plt.ylabel(list_ylabel[i],size=12)
    plt.xlabel('')
    #plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    
    if(i==0):
        plt.legend(fontsize=12,frameon=False)
    else:
        plt.legend().remove()
    ax.set_xlim(xmin=-0.5,xmax=3.5)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups_stat.png'.format(list_plots[i]))



cell3 = np.unique(df_allFullm['cell name 2'])
for m in range(len(cell3)):
    print(cell3[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name 2']==cell3[m]]))
   
###############################################################################
#
# plots
#
###############################################################################

plt.figure(figsize=figsize)
sns.lineplot(data=angleFil, x='name', y='angle',hue='cells',marker='o',markersize=10,palette=sns.color_palette("Set2")[0:3])
plt.legend(fontsize=12,frameon=False)
plt.ylabel('Circular mean angle ['r'$^\circ$]',fontsize=12)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_all.png')



plt.savefig(pathsave+'groups/'+'circ_mean_all.png')


plt.figure(figsize=figsize)
sns.lineplot(data=angleFil, x='name', y='circular variance',hue='cells',marker='o',markersize=10,palette=sns.color_palette("Set2")[0:3])
plt.legend(fontsize=20,frameon=False)
plt.ylabel('Circular mean variance',fontsize=12)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_var_all.png')


fig, axd = plt.subplot_mosaic("EFG;HIJ", figsize=(8.27,4))


sns.lineplot(data=angleFil, x='name', y='angle',hue='cells',marker='o',markersize=8,palette=sns.color_palette("Set2")[0:3],ax=axd['E'])


axd['E'].legend(loc='best',frameon=False)



plt.legend(fontsize=12,frameon=False)
#plt.xticks(rotation=45)
plt.tight_layout()
plt.ylabel('Circular mean angle ['r'$\degree$]',fontsize=12)
plt.xlabel('')









plt.close('all')


x = "cell name 3"
hue = "cell name"
hue_order = ['Young', 'Expanding', 'Mature']
order = ['<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

pairs  = [
    [('<3.22', 'Mature'), ('<3.22', 'Expanding')],
     [('<3.22', 'Mature'), ('<3.22', 'Young')],
     [('<3.22', 'Expanding'), ('<3.22', 'Young')],
     [('3.22-6.15', 'Mature'), ('3.22-6.15', 'Expanding')],
     [('3.22-6.15', 'Mature'), ('3.22-6.15', 'Young')],
     [('3.22-6.15', 'Expanding'), ('3.22-6.15', 'Young')],
     [('6.15-16.26', 'Mature'), ('6.15-16.26', 'Expanding')],
     [('6.15-16.26', 'Mature'), ('6.15-16.26', 'Young')],
     [('6.15-16.26', 'Expanding'), ('6.15-16.26', 'Young')],
     [('>16.26', 'Mature'), ('>16.26', 'Expanding')],
     [('>16.26', 'Mature'), ('>16.26', 'Young')],
     [('>16.26', 'Expanding'), ('>16.26', 'Young')],
     ]


list_plots = ['mean filament bendiness','mean intensity per length','mean filament movement']
list_ylabel = ['Mean bendiness ratio',r'Mean intensity per $\mu m$',r'Mean movement [$\mu m/s$]']


fig, axd = plt.subplot_mosaic("EFG;HIJ", figsize=(8.27,5))

plot=sns.lineplot(
    data=df_allFullm, 
    x="cell name 3", 
    y="mean filament angle", 
    hue="cell name", 
    marker='o', 
    markersize=8,
    n_boot=10000, 
    ci=95,
    ax=axd['E']
)
plot.legend_.set_title(None)
plot.legend(fontsize=8)

sns.boxplot(data=assFil, x='cells', y='Assortativity',showfliers = False, medianprops={"color": "coral"},palette=cmap,ax=axd['F'])
sns.stripplot(data=assFil, x="cells", y="Assortativity",s=3, dodge=True, ax=axd['F'],color='black')

sns.boxplot(data=assFil, x='cells', y='Cell density',showfliers = False,medianprops={"color": "coral"},palette=cmap,ax=axd['G'])
sns.stripplot(data=assFil, x="cells", y="Cell density",s=3, dodge=True, ax=axd['G'],color='black')


sns.boxplot(data=df_allFullm, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['H'])
annot = Annotator(axd['H'], pairs, data=df_allFullm, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['I'])
annot = Annotator(axd['I'], pairs, data=df_allFullm, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['J'])
annot = Annotator(axd['J'], pairs, data=df_allFullm, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()


axd['H'].legend(fontsize=8,frameon=False, loc="best")
axd['I'].legend().remove()
axd['J'].legend().remove()
axd['H'].set_xlim(xmin=-0.45,xmax=3.45)
axd['I'].set_xlim(xmin=-0.45,xmax=3.45)
axd['J'].set_xlim(xmin=-0.45,xmax=3.45)

adjust = 15
axd['E'].tick_params(axis='x', labelrotation=adjust)
#axd['F'].tick_params(axis='x', labelrotation=adjust)
#axd['G'].tick_params(axis='x', labelrotation=adjust)
axd['H'].tick_params(axis='x', labelrotation=adjust)
axd['I'].tick_params(axis='x', labelrotation=adjust)
axd['J'].tick_params(axis='x', labelrotation=adjust)

axd['E'].set_xlabel('')
axd['F'].set_xlabel("")
axd['G'].set_xlabel("")
axd['H'].set_xlabel('')
axd['I'].set_xlabel('')
axd['J'].set_xlabel('')

axd['E'].set_ylabel('Circular mean angle',fontsize=10)
axd['F'].set_ylabel("Assortativity",fontsize=10)
axd['G'].set_ylabel("Cell density",fontsize=10)

axd['H'].set_ylabel('Mean bendiness ratio',size=10)
axd['I'].set_ylabel(r'Mean intensity per $\mu m$',size=10)
axd['J'].set_ylabel(r'Mean movement [$\mu m/s$]',size=10)



for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')
 
plt.tight_layout()

plt.savefig(pathsave+'groups/'+'cotyledon_full_fig.pdf')
