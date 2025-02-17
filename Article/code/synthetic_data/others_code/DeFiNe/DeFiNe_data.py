#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 10:51:23 2024

@author: isabella
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pickle
import itertools
from scipy.optimize import linear_sum_assignment

overpath = 'define_this_path'

###############################################################################
#
# functions
#
###############################################################################

def new_key(edge_attributes):
    
    data_list = [(key[0], key[1], value) for key, value in edge_attributes.items()]
    array_data = np.array(data_list)

    new_rows = []

    # Iterate over each row in the NumPy array
    for row in array_data:
        start, end, values = row
        # Check if the value contains a semicolon
        if ';' in values:
            # Split the value by semicolon
            split_values = values.split(';')
            # Create a new row for each split value
            for value in split_values:
                new_rows.append([start, end, value])
        else:
            # If no semicolon, just add the original row
            new_rows.append(row)
    
    # Convert the list of new rows back to a NumPy array
    final_array = np.array(new_rows)

    return final_array

def identification_LAP(csv_path, pathGML, fm, savepathLines, box_size=500):
 
    selected_values = pd.read_csv(csv_path)
    
    frameT = selected_values.iloc[fm]['frame']
    if(frameT<101):
        nonoiseLinePos=100
    elif(frameT==500):
        nonoiseLinePos = 500
    else:
        nonoiseLinePos = ((100+(frameT-1))//100)*100
    pathNonoise='/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/noise{0}/{1}_lines_intermediate_nonoise_line_position.csv'.format(selected_values.iloc[fm]['line type'],nonoiseLinePos)

    df_Ori = pd.read_csv(pathNonoise)
    
    # filter away all values outside our frame of view
    df_Ori = df_Ori[(df_Ori['x_posR'] >= 0) & (df_Ori['x_posR'] < box_size)]
    df_Ori = df_Ori[(df_Ori['y_posR'] >= 0) & (df_Ori['y_posR'] < box_size)]

    df_timeseries = df_Ori[df_Ori['frame']==frameT].copy()
    
    graphTagg = nx.read_gml(pathGML+"img_{0}_sampling=1_overlap=1_quality=0_objective=0_angle=60_auto.gml".format(fm))
    posL = pd.read_pickle('/home/isabella/Documents/PLEN/dfs/others_code/DeFiNe/noise001/pos/{0}_posL.gpickle'.format(fm))
    
    results = []
    if(graphTagg!=0):
        graphTagg.edges(data=True)
        postLis = posL
        
        edge_attributes = nx.get_edge_attributes(graphTagg, 'auto')
        edgeAT = new_key(edge_attributes)
        marked = edgeAT[:,2]
        #tags = [int(marked[i].split('-')[0]) for i in range(len(marked))]
        
        
        filamentsLength = nx.get_edge_attributes(graphTagg, 'fdist')
        
        # Extract unique values from the dictionary values
        fValues = np.unique([int(marked[i].split('-')[0]) for i in range(len(marked))])
        trueSLst = df_timeseries['line_no'].unique()
        fLst = list(fValues)
        # costmatrix
        costM = np.zeros((len(trueSLst),len(fValues)))
        for nodeU in fValues:
            #find all node pos for edges marked as certain FS
            x_entries = [key for key in edge_attributes.keys() if nodeU.astype(str) in key]
            
            saved_values = []
            saved_values_length = []
            for z in range(len(x_entries)):
                
                node1 = postLis[int(x_entries[z][0])]-1
                node2 = postLis[int(x_entries[z][1])]-1
                
                node1x3,node1y3 = np.array([node1[0]-3, node1[0]-2, node1[0]-1, node1[0], node1[0]+1, node1[0]+2, node1[0]+3]), np.array([node1[1]-3, node1[1]-2, node1[1]-1, node1[1], node1[1]+1, node1[1]+2, node1[1]+3])
                node2x3,node2y3 = np.array([node2[0]-3, node2[0]-2, node2[0]-1, node2[0], node2[0]+1, node2[0]+2, node2[0]+3]), np.array([node2[1]-3, node2[1]-2, node2[1]-1, node2[1], node2[1]+1, node2[1]+2, node2[1]+3])
                
                node_valX = []
                xy_coordsX = []
                node_valY = []
                xy_coordsY = []
                for i in range(len(node1x3)):
                    for m in range(len(node1y3)):
                        # node 1
                        xdf = df_timeseries[(df_timeseries['x_positions'] == node1x3[i]) & (df_timeseries['y_positions'] == node1y3[m])]
                        #print(i,m,xdf)
                        if(not xdf.empty):
                            node_valX.append(xdf['line_no'].values[0])
                            xy_coordsX.append(xdf[['x_posR','y_posR']].values)
                            
                        # node 2
                        ydf = df_timeseries[(df_timeseries['x_positions'] == node2x3[i]) & (df_timeseries['y_positions'] == node2y3[m])]
                        #print(i,m,ydf)
                        if(not ydf.empty):
                            node_valY.append(ydf['line_no'].values[0])
                            xy_coordsY.append(ydf[['x_posR','y_posR']].values)
              
                # Find common elements     
                common_elements = list(set(node_valX).intersection(node_valY))
                
                
                saved_values.append(common_elements)
                saved_values_length.append(filamentsLength[x_entries[z]])
                
            # if no matching node pairs exist that overlap FS marks in timeseries
            is_empty = all(len(sublist) == 0 for sublist in saved_values)
            if(not is_empty):
                lengthEdges = np.asarray(saved_values_length)
                value_to_check = np.unique(list(itertools.chain.from_iterable(saved_values)))
                I_len_FilCoverage = np.zeros(len(value_to_check))
                I_trueLine = np.zeros(len(value_to_check))
                i_filLine = np.zeros(len(value_to_check))
                for ks,l in zip(value_to_check,range(len(value_to_check))):
                    lineLen = len(df_timeseries[(df_timeseries['line_no'] == ks)])
                    perEdge = lengthEdges/lineLen
                    # Calculate the total number of sublists
                    # Get indices of sublists that contain the specified value
                    indices = [index for index, sublist in enumerate(saved_values) if ks in sublist]
                    
                    I_len_FilCoverage[l] = np.sum(perEdge[indices])
                    i_filLine[l] = np.sum(lengthEdges[indices])
                    I_trueLine[l] = np.sum(lineLen)
                    trueI = np.where(trueSLst==ks)[0][0]
                    
                    costM[trueI,nodeU] = np.sum(perEdge[indices])
                    
        row_ind, col_ind = linear_sum_assignment(costM,maximize=True)    
        
        definedM = np.zeros(len(row_ind))
        deleteDef = []
        for n in range(len(row_ind)):
            list_true_line_len = len(df_timeseries[(df_timeseries['line_no'] == trueSLst[n])])
            #find correct matched filament
            x_entries = [key for key in edge_attributes.keys() if fLst[col_ind[n]].astype(str) in key]
            list_filament_line_len = sum(filamentsLength[key] for key in x_entries if key in filamentsLength)
            list_FilCoverage = costM[row_ind[n],col_ind[n]]
            definedM[n] = fLst[col_ind[n]]
            if(costM[row_ind[n],col_ind[n]]==0):
                deleteDef = np.append(deleteDef,int(n))
                bestM = None
                list_filament_line_len = 0
            else:
                bestM = fLst[col_ind[n]]

                
            results.append({'frame': frameT,
                            'true index': trueSLst[n],
                            'match index': bestM,
                            'FS_coverage': list_FilCoverage,
                            'FS_true_len': list_true_line_len,
                            'FS_found_len': list_filament_line_len,
                            'overlap ratio': list_FilCoverage
                            })
            
        if(len(deleteDef)!=0):
            definedM = np.delete(definedM,deleteDef.astype(int))
        main_list = list(set(fLst) - set(definedM))
        for x in range(len(main_list)):
            list_true_line_len = None
            #find correct matched filament
            x_entries = [key for key in edge_attributes.keys() if main_list[x].astype(str) in key]
            list_filament_line_len = sum(filamentsLength[key] for key in x_entries if key in filamentsLength)
            
            # find out if this defined filament oerlapped a true filament
            indices = np.where(np.asarray(fLst) == main_list[x])
            overlap = np.max(costM[:,indices])
            
            results.append({'frame': frameT,
                            'true index': None,
                            'match index': main_list[x],
                            'FS_coverage': 0,
                            'FS_true_len': list_true_line_len,
                            'FS_found_len': list_filament_line_len,
                            'overlap ratio': overlap
                            })
            
          
        results_df = pd.DataFrame(results)
        
        results_df.to_csv(savepathLines + '{0}_{1}_df_line_comparison.csv'.format(selected_values.iloc[fm]['line type'], frameT), index=False)  
    return 

###############################################################################
#
# run
#
###############################################################################

csv_path = overpath+'/noise001/data_test_other/which_frames_chosen_update.csv'
pathGML = overpath+"/others_code/DeFiNe/noise001/gml/"
savepathLines = overpath+'/dfs/others_code/DeFiNe/noise001/comparison/files/'

for ml in range(100):
    
    identification_LAP(csv_path, pathGML, ml, savepathLines, box_size=500)


gg = nx.read_gml(pathGML+"img_99_sampling=1_overlap=1_quality=0_objective=0_angle=60_auto.gml")
dftest = pd.read_csv(pathGML + 'img_99_sampling=1_overlap=1_quality=0_objective=0_angle=60_auto.csv')
dftest.columns

gg.edges()

list(gg.edges(data='auto'))
test = np.asarray(list(gg.edges(data='auto')))[:,2]
tags=[int(test[i].split('-')[0]) for i in range(len(test))]
len(np.unique(tags))

selected_values = pd.read_csv(overpath+'/noise001/data_test_other/which_frames_chosen_update.csv')

# check for nan
selected_values[np.isnan(selected_values['density groups image'])]
selected_values.columns

df_graft = pd.read_csv(overpath+'/noise001/pooled/' + 'pooled_data.csv')


dataUtils.identification_LAP(csv_list=csv_list,files_csv=files_csv,savepathLines=savepathLines,graph_path=graph_path,graph_list=graph_list,pos_list=pos_list,box_size=500)

