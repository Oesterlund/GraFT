#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:58:35 2023

@author: isabella
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle
import seaborn as sns

figsize = 9,6

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 

cmap=sns.color_palette("Set2")

size=20

plt.close('all')

###############################################################################
#
#
#
###############################################################################

pathsaveFig = '/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/figs/groups/'

plant=['p1','p2','p3','p4','p6','p7','p8']
tm = ['_top/','_mid/','_bot/']
pathsave2 = '/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/'

assFil=pd.DataFrame()
r_allC=[]
densityC=[]

for i in range(3):
    if(i==0):
        list_vals=[pathsave2+plant[0]+tm[0], pathsave2+plant[1]+tm[0], pathsave2+plant[2]+tm[0], pathsave2+plant[4]+tm[0], pathsave2+plant[5]+tm[0], pathsave2+plant[6]+tm[0]]
        name = 'Young'
    elif(i==1):
        list_vals=[pathsave2+plant[0]+tm[1],pathsave2+ plant[1]+tm[1], pathsave2+plant[2]+tm[1], pathsave2+plant[4]+tm[1], pathsave2+plant[5]+tm[1], pathsave2+plant[6]+tm[1]]
        name = 'Expanding'
    else:
        list_vals=[pathsave2+plant[0]+tm[2],pathsave2+ plant[1]+tm[2], pathsave2+plant[2]+tm[2], pathsave2+plant[3]+tm[2], pathsave2+plant[5]+tm[2], pathsave2+plant[6]+tm[2]]
        name = 'Mature'
    
    r_all=[]
    density = []
    
    for c in range(len(list_vals)):
        file = open(list_vals[c]+'tagged_graph.gpickle','rb')
        tagged_graph = pickle.load(file)
        pd_fil_info = pd.read_csv(list_vals[c]+'tracked_filaments_info.csv')
        r_db=[]
        dens_db = []
        for i in range(len(tagged_graph)):
            r_db=np.append(r_db,nx.degree_assortativity_coefficient(tagged_graph[i]))
        
        dens_db = np.unique(pd_fil_info['filament density'])
        density=np.append(density,np.mean(dens_db))
        r_all=np.append(r_all,np.mean(r_db))

    
    r_allC = np.append(r_allC,r_all)#.reshape((4, 4))
    densityC = np.append(densityC,density)#.reshape((4, 4))

r_C = r_allC.reshape((3,6))
d_C = densityC.reshape((3,6))
name_cell = ['Young','Expanding','Mature']

for i in range(len(name_cell)):
    dataang = {'Assortativity' : r_C[i],'Cell density':d_C[i],'cells': name_cell[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    assFil = pd.concat(([assFil,frame]),ignore_index=True)

###################
# plots
#
'''
plt.figure(figsize=figsize)
sns.scatterplot(data=assFil, x='cells', y='Assortativity',hue='cells',marker='o',alpha=0.7,s=200,palette=cmap[0:3])
sns.scatterplot(x='cells', y='Assortativity',data=assFil[assFil['cells']=='Upper'].groupby('cells', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="+", label="Average upper",color='k')
sns.scatterplot(x='cells', y='Assortativity',data=assFil[assFil['cells']=='Middle'].groupby('cells', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="*", label="Average middle",c='k')
sns.scatterplot(x='cells', y='Assortativity',data=assFil[assFil['cells']=='Bottom'].groupby('cells', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="o", label="Average bottom",c='k')

plt.legend(fontsize=size)
plt.ylabel('Assortativity',fontsize=25)
#plt.ylim(-0.3,0.3)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsaveFig+'assortativity.png')
'''
    
plt.figure(figsize=figsize)
ax = sns.boxplot(data=assFil, x='cells', y='Assortativity',showfliers = False,
            medianprops={"color": "coral"},palette=cmap)
sns.stripplot(data=assFil, x="cells", y="Assortativity",s=10, dodge=True, ax=ax,color='black')
plt.xlabel('')
plt.legend(fontsize=size)
plt.ylabel('Assortativity',fontsize=25)
plt.legend().remove()
plt.tight_layout()
plt.savefig(pathsaveFig+'assortativity2.png')

    
'''

plt.figure(figsize=(12,8))
sns.scatterplot(data=assFil, x='cells', y='Cell density',hue='cells',marker='o',alpha=0.7,s=200)#,palette=cmap)
sns.scatterplot(x='cells', y='Cell density',data=assFil[assFil['cells']=='Upper'].groupby('cells', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="+", label="Average upper",c='k')
sns.scatterplot(x='cells', y='Cell density',data=assFil[assFil['cells']=='Middle'].groupby('cells', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="*", label="Average middle",c='k')
sns.scatterplot(x='cells', y='Cell density',data=assFil[assFil['cells']=='Bottom'].groupby('cells', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="o", label="Average bottom",c='k')

plt.legend(fontsize=size)
plt.ylabel('Cell density',fontsize=25)
#plt.ylim(0.01,0.2)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsaveFig+'cell_density.png')
'''

plt.figure(figsize=figsize)
ax = sns.boxplot(data=assFil, x='cells', y='Cell density',showfliers = False,
            medianprops={"color": "coral"},palette=cmap)
sns.stripplot(data=assFil, x="cells", y="Cell density",s=10, dodge=True, ax=ax,color='black')
plt.xlabel('')
plt.legend(fontsize=size)
plt.ylabel('Cell density',fontsize=25)
plt.legend().remove()
plt.tight_layout()
plt.savefig(pathsaveFig+'cell_density2.png')
