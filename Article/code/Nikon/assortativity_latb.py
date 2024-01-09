#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:42:38 2023

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle
import seaborn as sns
import astropy.stats
from astropy import units as u

figsize = 9,6

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 

sizeL=25
plt.close('all')

cmap=sns.color_palette("Set2")
###############################################################################
#
#
#
###############################################################################

def circ_stat(DataSet):

    #data = np.asarray(pd_fil_info['filament angle'])*u.deg
    #weight = pd_fil_info['filament length']
    data = DataSet["filament angle"]*u.deg
    weight = DataSet['filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))*180./np.pi
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))
        
    return mean_angle,var_val

###############################################################################
#
#
#
###############################################################################


assFil=pd.DataFrame()
for i in range(2):
    if(i==0):
        pathsave = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/"
        pathsaveFig = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/figs/'
        name = 'Light'
        
    else:
        pathsave = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/"
        pathsaveFig = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/figs/'
        name = 'Dark'
    plant=['control/','latB_0-10min/','latB_10-20min/','latB_20-30min/']
    tm = ['cell1/','cell2/','cell3/','cell4/']
    
    
    r_all=[]
    density = []
    angle = []
    angleVar = []
    
    for c in range(len(plant)):
        for m in range(len(tm)):
            file = open(pathsave+plant[c]+tm[m]+'tagged_graph.gpickle','rb')
            tagged_graph = pickle.load(file)
            pd_fil_info = pd.read_csv(pathsave+plant[c]+tm[m]+'tracked_filaments_info.csv')
            r_db=[]
            dens_db = []
            for i in range(len(tagged_graph)):
                r_db=np.append(r_db,nx.degree_assortativity_coefficient(tagged_graph[i]))
            
            dens_db = np.unique(pd_fil_info['filament density'])
            
            angleD,varD = circ_stat(pd_fil_info)
            
            density=np.append(density,np.mean(dens_db))
            r_all=np.append(r_all,np.mean(r_db))
            angle=np.append(angle,angleD)
            angleVar=np.append(angleVar,varD)
    
    r_allC = r_all.reshape((4, 4))
    densityC = density.reshape((4, 4))
    angleC = angle.reshape((4,4))
    angleVarC = angleVar.reshape((4,4))
    
    
    name_cell = ['Control','0-10 mins','10-20 mins','20-30 mins']
    
    for i in range(len(name_cell)):
        name_cellN = name_cell[i]+' {0}'.format(name)
        dataang = {'Assortativity' : r_allC[i],'Cell density':densityC[i], 'Mean circular angle': angleC[i],'Mean angle var':angleVarC[i],'cells': name_cellN,'Treatment':name_cell[i],'Type':name}
        frame = pd.DataFrame.from_dict(dataang)
        
        assFil = pd.concat(([assFil,frame]),ignore_index=True)
    
###################
# plots
#

plt.figure(figsize=figsize)
sns.scatterplot(data=assFil, x='Treatment', y='Assortativity',hue='Type',marker='o',alpha=0.7,s=200,palette=cmap[0:2])
sns.scatterplot(x='Treatment', y='Assortativity',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='black')
sns.scatterplot(x='Treatment', y='Assortativity',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='black')
sns.lineplot(x='Treatment', y='Assortativity',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Assortativity'].mean(),color=(0.4, 0.7607843137254902, 0.6470588235294118),alpha=0.5)
sns.lineplot(x='Treatment', y='Assortativity',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Assortativity'].mean(),color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),alpha=.5)


plt.legend(fontsize=20,frameon=False)
plt.ylabel('Assortativity',fontsize=sizeL)
plt.ylim(-0.3,0.3)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsaveFig+'groups/'+'assortativity.png')

    
    
plt.figure(figsize=figsize)
sns.scatterplot(data=assFil, x='Treatment', y='Cell density',hue='Type',marker='o',alpha=0.7,s=200,palette=cmap[0:2])
sns.scatterplot(x='Treatment', y='Cell density',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='black')
sns.scatterplot(x='Treatment', y='Cell density',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='black')
sns.lineplot(x='Treatment', y='Cell density',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Cell density'].mean(),color=(0.4, 0.7607843137254902, 0.6470588235294118),alpha=0.5)
sns.lineplot(x='Treatment', y='Cell density',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Cell density'].mean(),color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),alpha=.5)

plt.legend(fontsize=20,frameon=False)
plt.ylabel('Cell density',fontsize=sizeL)
plt.ylim(0.01,0.2)
plt.xlabel('')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(pathsaveFig+'groups/'+'cell_density.png')

    
plt.figure(figsize=figsize)
sns.scatterplot(data=assFil, x='Treatment', y='Mean circular angle',hue='Type',marker='o',alpha=0.7,s=200,palette=cmap[0:2])
sns.scatterplot(x='Treatment', y='Mean circular angle',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Mean circular angle'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='black')
sns.scatterplot(x='Treatment', y='Mean circular angle',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Mean circular angle'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='black')
sns.lineplot(x='Treatment', y='Mean circular angle',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Mean circular angle'].mean(),color=(0.4, 0.7607843137254902, 0.6470588235294118),alpha=0.5)
sns.lineplot(x='Treatment', y='Mean circular angle',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Mean circular angle'].mean(),color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),alpha=.5)

plt.legend(fontsize=30).remove()
plt.ylabel('Circular mean angle ['r'$\degree$]',fontsize=sizeL)
#plt.ylim(0.01,0.2)
plt.xlabel('')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(pathsaveFig+'groups/'+'cell_angle.png')


plt.figure(figsize=figsize)
sns.scatterplot(data=assFil, x='Treatment', y='Mean angle var',hue='Type',marker='o',alpha=0.7,s=200,palette=cmap[0:2])
sns.scatterplot(x='Treatment', y='Mean angle var',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Mean angle var'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='black')
sns.scatterplot(x='Treatment', y='Mean angle var',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Mean angle var'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='black')
sns.lineplot(x='Treatment', y='Mean angle var',data=assFil[assFil['Type']=='Light'].groupby('Treatment', as_index=False)['Mean angle var'].mean(),color=(0.4, 0.7607843137254902, 0.6470588235294118),alpha=0.5)
sns.lineplot(x='Treatment', y='Mean angle var',data=assFil[assFil['Type']=='Dark'].groupby('Treatment', as_index=False)['Mean angle var'].mean(),color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),alpha=.5)

plt.legend(fontsize=30).remove()
plt.ylabel('Circular variance',fontsize=sizeL)
#plt.ylim(0.01,0.2)
plt.xlabel('')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(pathsaveFig+'groups/'+'cell_var.png')
