#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:56:12 2023

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

sizeL=25
plt.close('all')

cmap=sns.color_palette("Set2")

plt.close('all')

###############################################################################
#
#
#
###############################################################################


assFil=pd.DataFrame()
pathsave = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/"
pathsaveFig = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/figs/'


plant=['DMSO/','DSF/','flg22/']
tm = ['cell1/','cell2/','cell3/','cell4/','cell5/']

r_all=[]
density = []

for c in range(len(plant)):
    print(plant[c])
    for m in range(4):
        if(plant[c]=='DSF/'):
            m=m+1
        if((plant[c]=='flg22/') and (m==3)):
            m=m+1
        file = open(pathsave+plant[c]+tm[m]+'tagged_graph.gpickle','rb')
        tagged_graph = pickle.load(file)
        pd_fil_info = pd.read_csv(pathsave+plant[c]+tm[m]+'tracked_filaments_info.csv')
        r_db=[]
        dens_db = []
        for i in range(len(tagged_graph)):
            r_db=np.append(r_db,nx.degree_assortativity_coefficient(tagged_graph[i]))
            
        group = pd_fil_info.groupby('frame number')
        dens_db = np.mean(group.apply(lambda x: x['filament density'].unique()))
        
        print(plant[c],np.mean(dens_db))
        density=np.append(density,np.mean(dens_db))
        r_all=np.append(r_all,np.mean(r_db))


r_allC = r_all.reshape((3, 4))
densityC = density.reshape((3, 4))


name_cell = ['Control','DSF','flg22']

for i in range(len(name_cell)):
    dataang = {'Assortativity' : r_allC[i],'Cell density':densityC[i],'Treatment':name_cell[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    assFil = pd.concat(([assFil,frame]),ignore_index=True)

###################
# plots
#

plt.figure(figsize=figsize)
sns.scatterplot(data=assFil, x='Treatment', y='Assortativity',hue='Treatment',marker='o',alpha=0.7,s=200,palette=cmap[0:3])
sns.scatterplot(x='Treatment', y='Assortativity',data=assFil[assFil['Treatment']=='Control'].groupby('Treatment', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='k')
sns.scatterplot(x='Treatment', y='Assortativity',data=assFil[assFil['Treatment']=='DSF'].groupby('Treatment', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='k')
sns.scatterplot(x='Treatment', y='Assortativity',data=assFil[assFil['Treatment']=='flg22'].groupby('Treatment', as_index=False)['Assortativity'].mean(), s=250, alpha=1.,marker="*", label="Average dark",color='k')

plt.legend(fontsize=20,frameon=False)
plt.ylabel('Assortativity',fontsize=30)
#plt.ylim(-0.3,0.3)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsaveFig+'groups/'+'assortativity.png')

    

plt.figure(figsize=figsize)
sns.scatterplot(data=assFil, x='Treatment', y='Cell density',hue='Treatment',marker='o',alpha=0.7,s=200,palette=cmap[0:3])
sns.scatterplot(x='Treatment', y='Cell density',data=assFil[assFil['Treatment']=='Control'].groupby('Treatment', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="+", label="Average control",color='k')
sns.scatterplot(x='Treatment', y='Cell density',data=assFil[assFil['Treatment']=='DSF'].groupby('Treatment', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="*", label="Average DSF",color='k')
sns.scatterplot(x='Treatment', y='Cell density',data=assFil[assFil['Treatment']=='flg22'].groupby('Treatment', as_index=False)['Cell density'].mean(), s=250, alpha=1.,marker="o", label="Average flg22",color='k')

plt.legend(fontsize=20)
plt.ylabel('Cell density',fontsize=sizeL)
plt.ylim(0.095,0.21)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsaveFig+'groups/'+'cell_density.png')



plt.figure(figsize=figsize)
ax = sns.boxplot(data=assFil, x='Treatment', y='Cell density',showfliers = False,
            medianprops={"color": "coral"},palette=cmap)
sns.stripplot(data=assFil, x="Treatment", y="Cell density",s=10, dodge=True, ax=ax,color='black')
plt.xlabel('')
plt.legend(fontsize=20,frameon=False)
plt.ylabel('Cell density',fontsize=sizeL)
plt.legend().remove()
plt.tight_layout()
plt.savefig(pathsaveFig+'groups/'+'cell_density2.png')

np.mean(densityC,axis=1)
