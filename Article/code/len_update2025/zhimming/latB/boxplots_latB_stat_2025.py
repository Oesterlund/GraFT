#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:16:23 2025

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns

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

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/zhimming/latB/figs/'
pixelS = 0.065

figsize=(8.27/2, 2.5)

###############################################################################
#
# import files
#
###############################################################################

df_all_light = pd.read_csv(pathsave+'df_all_light.csv', encoding='utf-8')
df_allFullm_light = pd.read_csv(pathsave+'df_allFullm_light.csv', encoding='utf-8')

df_all_dark = pd.read_csv(pathsave+'df_all_dark.csv', encoding='utf-8')
df_allFullm_dark = pd.read_csv(pathsave+'df_allFullm_dark.csv', encoding='utf-8')

angleFil_light = pd.read_csv(pathsave+'df_angles_light.csv', encoding='utf-8')
angleFil_dark = pd.read_csv(pathsave+'df_angles_dark.csv', encoding='utf-8')

#angleFil2 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/figs/df_angles.csv', encoding='utf-8')
assFil = pd.read_csv(pathsave+'df_assFil.csv', encoding='utf-8')

angleFil_light['sun']='light'
angleFil_dark['sun']='dark'
angleFil = pd.concat([angleFil_light,angleFil_dark], ignore_index=True).dropna()

###############################################################################
#
# plotting
#
###############################################################################

df_allFullm_light['LD'] = 'Light'
df_allFullm_dark['LD'] = 'Dark'
df_allFullm = pd.concat([df_allFullm_light, df_allFullm_dark], ignore_index=True)

df_allFullm['cell name X'] = df_allFullm['cell name']
df_allFullm['cell name X'][df_allFullm['cell name']=='0-10 mins']='0-10'
df_allFullm['cell name X'][df_allFullm['cell name']=='10-20 mins']='10-20'
df_allFullm['cell name X'][df_allFullm['cell name']=='20-30 mins']='20-30'

x = "cell name 3"
hue = "cell name"
hue_order = ['Control', '0-10 mins', '10-20 mins','20-30 mins']
order = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

pairsLight  = [
    [('<3.22', 'Control'), ('<3.22', '0-10 mins')],
     [('<3.22', 'Control'), ('<3.22', '10-20 mins')],
     [('<3.22', 'Control'), ('<3.22', '20-30 mins')],
     [('<3.22', '0-10 mins'), ('<3.22', '10-20 mins')],
     [('<3.22', '0-10 mins'), ('<3.22', '20-30 mins')],
     [('<3.22', '10-20 mins'), ('<3.22', '20-30 mins')],
     
     [('3.22-6.15', 'Control'), ('3.22-6.15', '0-10 mins')],
     [('3.22-6.15', 'Control'), ('3.22-6.15', '10-20 mins')],
     [('3.22-6.15', 'Control'), ('3.22-6.15', '20-30 mins')],
     [('3.22-6.15', '0-10 mins'), ('3.22-6.15', '10-20 mins')],
     [('3.22-6.15', '0-10 mins'), ('3.22-6.15', '20-30 mins')],
     [('3.22-6.15', '10-20 mins'), ('3.22-6.15', '20-30 mins')],

     [('6.15-16.26', 'Control'), ('6.15-16.26', '0-10 mins')],
     [('6.15-16.26', 'Control'), ('6.15-16.26', '10-20 mins')],
     [('6.15-16.26', 'Control'), ('6.15-16.26', '20-30 mins')],
     [('6.15-16.26', '0-10 mins'), ('6.15-16.26', '10-20 mins')],
     [('6.15-16.26', '0-10 mins'), ('6.15-16.26', '20-30 mins')],
     [('6.15-16.26', '10-20 mins'), ('6.15-16.26', '20-30 mins')],
     
     [('>16.26', 'Control'), ('>16.26', '0-10 mins')],
     [('>16.26', 'Control'), ('>16.26', '10-20 mins')],
     [('>16.26', 'Control'), ('>16.26', '20-30 mins')],
     [('>16.26', '0-10 mins'), ('>16.26', '10-20 mins')],
     [('>16.26', '0-10 mins'), ('>16.26', '20-30 mins')],
     [('>16.26', '10-20 mins'), ('>16.26', '20-30 mins')],
     ]


pairsDark  = [
    [('<3.22', 'Control'), ('<3.22', '0-10 mins')],
     [('<3.22', 'Control'), ('<3.22', '10-20 mins')],
     [('<3.22', 'Control'), ('<3.22', '20-30 mins')],
     [('<3.22', '0-10 mins'), ('<3.22', '10-20 mins')],
     [('<3.22', '0-10 mins'), ('<3.22', '20-30 mins')],
     [('<3.22', '10-20 mins'), ('<3.22', '20-30 mins')],
     
     [('3.22-6.15', 'Control'), ('3.22-6.15', '0-10 mins')],
     [('3.22-6.15', 'Control'), ('3.22-6.15', '10-20 mins')],
     [('3.22-6.15', 'Control'), ('3.22-6.15', '20-30 mins')],
     [('3.22-6.15', '0-10 mins'), ('3.22-6.15', '10-20 mins')],
     [('3.22-6.15', '0-10 mins'), ('3.22-6.15', '20-30 mins')],
     [('3.22-6.15', '10-20 mins'), ('3.22-6.15', '20-30 mins')],

     [('6.15-16.26', 'Control'), ('6.15-16.26', '0-10 mins')],
     [('6.15-16.26', 'Control'), ('6.15-16.26', '10-20 mins')],
     [('6.15-16.26', 'Control'), ('6.15-16.26', '20-30 mins')],
     [('6.15-16.26', '0-10 mins'), ('6.15-16.26', '10-20 mins')],
     [('6.15-16.26', '0-10 mins'), ('6.15-16.26', '20-30 mins')],
     [('6.15-16.26', '10-20 mins'), ('6.15-16.26', '20-30 mins')],
     
     [('>16.26', 'Control'), ('>16.26', '0-10 mins')],
     [('>16.26', 'Control'), ('>16.26', '10-20 mins')],
     #[('>19.5', 'Control'), ('>19.5', '20-30 mins')],
     [('>16.26', '0-10 mins'), ('>16.26', '10-20 mins')],
     #[('>19.5', '0-10 mins'), ('>19.5', '20-30 mins')],
     #[('>19.5', '10-20 mins'), ('>19.5', '20-30 mins')],
     ]


plt.close('all')
fig, axd = plt.subplot_mosaic("IJK;LMN;OPQ", figsize=(8.27,7.5))

######
# for light
sns.boxplot(data=df_allFullm_light, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['I'])
annot = Annotator(axd['I'], pairsLight, data=df_allFullm_light, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm_light, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['J'])
annot = Annotator(axd['J'], pairsLight, data=df_allFullm_light, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm_light, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['K'])
annot = Annotator(axd['K'], pairsLight, data=df_allFullm_light, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

######
# for dark

sns.boxplot(data=df_allFullm_dark, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['L'])
annot = Annotator(axd['L'], pairsDark, data=df_allFullm_dark, x=x, y='mean filament movement', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm_dark, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['M'])
annot = Annotator(axd['M'], pairsDark, data=df_allFullm_dark, x=x, y='mean filament bendiness', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

sns.boxplot(data=df_allFullm_dark, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
            medianprops={"color": "coral"},palette=cmap,ax=axd['N'])
annot = Annotator(axd['N'], pairsDark, data=df_allFullm_dark, x=x, y='mean intensity per length', order=order, hue=hue, hue_order=hue_order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=9,hide_non_significant=True)
annot.apply_test()
annot.annotate()

#########
# both

plot3 = sns.lineplot(data=assFil, 
                     x='Treatment', 
                     y='Cell density',
                     hue='Type',
                     marker='o', 
                     markersize=8,
                     n_boot=10000, 
                     ci=95,
                     ax=axd['O'])
plot3.legend_.set_title(None)
plot3.legend(fontsize=8,frameon=False, loc="best")


plot2=sns.lineplot(
    data=angleFil, 
    x="cells", 
    y="angle", 
    hue="sun", 
    marker='o', 
    markersize=8,
    n_boot=10000, 
    ci=95,
    ax=axd['P']
)
plot2.legend_.remove()

plot1=sns.lineplot(
    data=angleFil, 
    x="cells", 
    y="circular variance", 
    hue="sun", 
    marker='o', 
    markersize=8,
    n_boot=10000, 
    ci=95,
    ax=axd['Q']
)
plot1.legend_.remove()


axd['I'].legend(fontsize=6,frameon=False, loc="best")
axd['J'].legend().remove()
axd['K'].legend().remove()
axd['I'].set_xlim(xmin=-0.45,xmax=3.45)
axd['J'].set_xlim(xmin=-0.45,xmax=3.45)
axd['K'].set_xlim(xmin=-0.45,xmax=3.45)

axd['L'].legend(fontsize=6,frameon=False, loc="best").remove()
axd['M'].legend().remove()
axd['N'].legend().remove()
axd['L'].set_xlim(xmin=-0.45,xmax=3.25)
axd['M'].set_xlim(xmin=-0.45,xmax=3.25)
axd['N'].set_xlim(xmin=-0.45,xmax=3.25)

adjust = 15
axd['I'].tick_params(axis='x', labelrotation=adjust)
axd['J'].tick_params(axis='x', labelrotation=adjust)
axd['K'].tick_params(axis='x', labelrotation=adjust)
axd['L'].tick_params(axis='x', labelrotation=adjust)
axd['M'].tick_params(axis='x', labelrotation=adjust)
axd['N'].tick_params(axis='x', labelrotation=adjust)
axd['O'].tick_params(axis='x', labelrotation=adjust)
axd['P'].tick_params(axis='x', labelrotation=adjust)
axd['Q'].tick_params(axis='x', labelrotation=adjust)

axd['I'].set_xlabel('')
axd['J'].set_xlabel("")
axd['K'].set_xlabel("")

axd['L'].set_xlabel('')
axd['M'].set_xlabel("")
axd['N'].set_xlabel("")

axd['O'].set_xlabel('')
axd['P'].set_xlabel("")
axd['Q'].set_xlabel("")

axd['I'].set_ylabel(r'Mean movement [$\mu m/s$]',size=10)
axd['J'].set_ylabel('Mean bendiness ratio',size=10)
axd['K'].set_ylabel(r'Mean intensity per $\mu m$',size=10)

axd['L'].set_ylabel(r'Mean movement [$\mu m/s$]',size=10)
axd['M'].set_ylabel('Mean bendiness ratio',size=10)
axd['N'].set_ylabel(r'Mean intensity per $\mu m$',size=10)

axd['O'].set_ylabel("Cell density",fontsize=10)
axd['P'].set_ylabel('Circular mean angle',fontsize=10)
axd['Q'].set_ylabel("Circular mean variance",fontsize=10)



for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')
    
keys_to_label = {
    'J': 'Light',
    'M': 'Dark',
    'P': 'Light & Dark'
}

for key, label in keys_to_label.items():
    ax = axd[key]
    ax.text(0.5, 1.1, label, transform=ax.transAxes, 
            ha='center', size=12, weight='bold')
#axd['J'].text(0.5, 1.1, 'Light', transform=ax.transAxes, size=12, weight='bold',ha='center')
#axd['M'].text(0.5, 1.1, 'Dark', transform=ax.transAxes, size=12, weight='bold',ha='center')
#axd['P'].text(0.5, 1.1, 'Light & Dark', transform=ax.transAxes, size=12, weight='bold',ha='center')

plt.tight_layout()

plt.savefig(pathsave+'groups/'+'latB_full_fig.pdf')
