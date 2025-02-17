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
import scienceplots
import argparse
import matplotlib.ticker as ticker


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

figsize=(8.27/2, 2.5)

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/cotyledon/figs/'

###############################################################################
#
# functions
#
###############################################################################

def find_degree_by_tags(G):
    # Step 1: Filter edges with 'filament dangling' = 1
    relevant_edges = [(u, v, k, d) for u, v, k, d in G.edges(data=True, keys=True) if d.get('filament dangling') == 1]
    
    # Step 2: Group edges by 'tags'
    edges_by_tags = {}
    for u, v, k, d in relevant_edges:
        tag = d.get('tags')
        if tag not in edges_by_tags:
            edges_by_tags[tag] = []
        edges_by_tags[tag].append((u, v))  # Store only the node pairs (u, v)
    
    # Step 3: Calculate the highest and lowest degree for each 'tags' group
    highest_degree_by_tag = {}
    lowest_degree_by_tag = {}
    
    for tag, edges in edges_by_tags.items():
        # Collect all nodes involved in the edges with the same 'tags'
        nodes_in_tag_group = set()
        for u, v in edges:
            nodes_in_tag_group.update([u, v])
        
        if len(nodes_in_tag_group) == 0:
            # No nodes in the group, set degree to 0
            highest_degree_by_tag[tag] = 0
            lowest_degree_by_tag[tag] = 0
        else:
            # Calculate the degree of each node in the original graph
            degrees = [G.degree(node) for node in nodes_in_tag_group]
            
            # Store the highest and lowest degree for the current tag group
            highest_degree_by_tag[tag] = max(degrees)
            lowest_degree_by_tag[tag] = min(degrees)
    
    # Step 4: Convert to pandas DataFrame
    df_degree = pd.DataFrame.from_dict(highest_degree_by_tag, orient='index', columns=['Max Degree'])

    # Convert lowest degree dictionary to DataFrame column
    df_degree['Min Degree'] = pd.Series(lowest_degree_by_tag)

    # Reset the index to have 'tags' as a column instead of index
    df_degree = df_degree.reset_index().rename(columns={'index': 'Tags'})
    
    return df_degree['Max Degree'].mean(), df_degree['Min Degree'].mean()

###############################################################################
#
#
#
###############################################################################

pathsaveFig = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/cotyledon/figs/groups/'

plant=['p1','p2','p3','p4','p6','p7','p8']
tm = ['_top/','_mid/','_bot/']
pathsave2 = '/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/'

assFil=pd.DataFrame()
r_allC=[]
densityC=[]
dMax_allC = []
dMin_allC = []

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
    dMax_all = []
    dMin_all = []
    
    for c in range(len(list_vals)):
        file = open(list_vals[c]+'tagged_graph.gpickle','rb')
        tagged_graph = pickle.load(file)
        pd_fil_info = pd.read_csv(list_vals[c]+'tracked_filaments_info.csv')
        r_db=[]
        dens_db = []
        danglingMax = []
        danglingMin = []
        for i in range(len(tagged_graph)):
            r_db=np.append(r_db,nx.degree_assortativity_coefficient(tagged_graph[i]))
            
            dMax, dMin = find_degree_by_tags(tagged_graph[i])
            danglingMax = np.append(danglingMax,dMax)
            danglingMin = np.append(danglingMin,dMin)
            
        dens_db = np.unique(pd_fil_info['filament density'])
        density=np.append(density,np.mean(dens_db))
        r_all=np.append(r_all,np.mean(r_db))
        dMax_all = np.append(dMax_all,np.mean(danglingMax))
        dMin_all = np.append(dMin_all,np.mean(danglingMin))
    
    r_allC = np.append(r_allC,r_all)#.reshape((4, 4))
    densityC = np.append(densityC,density)#.reshape((4, 4))
    
    dMax_allC = np.append(dMax_allC,dMax_all)
    dMin_allC = np.append(dMin_allC,dMin_all)

r_C = r_allC.reshape((3,6))
d_C = densityC.reshape((3,6))
degreeMax = dMax_allC.reshape((3,6))
degreeMin = dMin_allC.reshape((3,6))

name_cell = ['Young','Expanding','Mature']

for i in range(len(name_cell)):
    dataang = {'Assortativity' : r_C[i],'Cell density':d_C[i],'cells': name_cell[i],'Node degree max':degreeMax[i],'Node degree min':degreeMin[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    assFil = pd.concat(([assFil,frame]),ignore_index=True)


assFil.to_csv(pathsave+'df_assFil.csv')


###################
# plots
#

plt.figure(figsize=figsize)
ax = sns.boxplot(data=assFil, x='cells', y='Assortativity',showfliers = False,
            medianprops={"color": "coral"},palette=cmap)
sns.stripplot(data=assFil, x="cells", y="Assortativity",s=5, dodge=True, ax=ax,color='black')
plt.xlabel('')
plt.legend(fontsize=12)
plt.ylabel('Assortativity',fontsize=12)
plt.legend().remove()
plt.tight_layout()
plt.savefig(pathsaveFig+'assortativity2.png')

plt.figure(figsize=figsize)
ax = sns.boxplot(data=assFil, x='cells', y='Cell density',showfliers = False,
            medianprops={"color": "coral"},palette=cmap)
sns.stripplot(data=assFil, x="cells", y="Cell density",s=10, dodge=True, ax=ax,color='black')
plt.xlabel('')
plt.legend(fontsize=12)
plt.ylabel('Cell density',fontsize=12)
plt.legend().remove()
plt.tight_layout()
plt.savefig(pathsaveFig+'cell_density2.png')



