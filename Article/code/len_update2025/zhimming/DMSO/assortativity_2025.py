#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 22:26:47 2025

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

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/zhimming/DMSO/figs/'

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


tm = ['cell1/','cell2/','cell3/','cell4/','cell5/']

pathsave_DMSO = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO/"
pathsave_DSF = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DSF/"
pathsave_flg22 = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/flg22/"

pathRN  = [pathsave_DMSO, pathsave_DSF, pathsave_flg22]

assFil=pd.DataFrame()
r_allC=[]
densityC=[]
dMax_allC = []
dMin_allC = []

for i in range(3):
    if(i==0):
        list_vals=[pathRN[0]+tm[0], pathRN[0]+tm[1], pathRN[0]+tm[2], pathRN[0]+tm[3]]
        name = 'DMSO'
    elif(i==1):
        list_vals=[pathRN[1]+tm[1], pathRN[1]+tm[2], pathRN[1]+tm[3], pathRN[1]+tm[4]]
        name = 'DSF'
    else:
        list_vals=[pathRN[2]+tm[0],pathRN[2]+tm[1], pathRN[2]+tm[2], pathRN[2]+tm[4]]
        name = 'flg22'
    
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

r_C = r_allC.reshape((3,4))
d_C = densityC.reshape((3,4))
degreeMax = dMax_allC.reshape((3,4))
degreeMin = dMin_allC.reshape((3,4))

name_cell = ['DMSO','DSF','flg22']

for i in range(len(name_cell)):
    dataang = {'Assortativity' : r_C[i],'Cell density':d_C[i],'cells': name_cell[i],'Node degree max':degreeMax[i],'Node degree min':degreeMin[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    assFil = pd.concat(([assFil,frame]),ignore_index=True)


assFil.to_csv(pathsave+'df_assFil.csv')



