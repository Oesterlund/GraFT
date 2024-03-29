#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 08:57:59 2023

@author: pgf840
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
import networkx as nx
import pandas as pd
from collections import Counter
import pickle

path="/Users/pgf840/Documents/PLEN/tracking-filaments/dfs/GraFT"
os.chdir(path)

import utilsF

pathsave = "/Users/pgf840/Documents/PLEN/tracking-filaments/dfs/GraFT/"

plt.close('all')

sigma = 1.0                                                                     # tubeness filter width
small = 50.0                                                                    # cluster removal  

###############################################################################
#
# function to create all
#
###############################################################################

def create_all(pathsave,img_o,maskDraw,size,eps,thresh_top,sigma,small,angleA,overlap,max_cost,name_cell):
    # check if folders exists:
    path_check = [pathsave+'n_graphs',pathsave+'circ_stat',pathsave+'mov',pathsave+'plots']
    for i in range(len(path_check)):
        
        if not os.path.exists(path_check[i]):
            os.makedirs(path_check[i])
    
    graph_s = [0]*len(img_o)
    posL = [0]*len(img_o)
    imgSkel = [0]*len(img_o)
    imgAF = [0]*len(img_o)
    imgBl = [0]*len(img_o)
    imF = [0]*len(img_o)
    mask = [0]*len(img_o)
    df_pos = [0]*len(img_o)
    graphD = [0]*len(img_o)
    lgG = [0]*len(img_o)
    lgG_V = [0]*len(img_o)
    graphTagg = [0]*len(img_o)
    no_filaments = [0]*len(img_o)
    
    M,N,P = (img_o.shape)
    imgP=np.zeros((M,N+2,P+2))
    
    for m in range(len(img_o)):
        imgP[m] = np.pad(img_o[m], 1, 'constant')
        
    for q in range(len(imgP)):
        
        print(q)
        # 0) create graph
        graph_s[q], posL[q], imgSkel[q], imgAF[q], imgBl[q],imF[q],mask[q],df_pos[q] = utilsF.creategraph(imgP[q],size=size,eps=eps,thresh_top=thresh_top,sigma=sigma,small=small)
        utilsF.draw_graph(imgSkel[q],graph_s[q],posL[q],"untagged graph")
        #plt.savefig(pathsave+'n_graphs/untaggedgraph{0}.png'.format(q))
        # 1) find all dangling edges and mark them
        graphD[q] = utilsF.dangling_edges(graph_s[q].copy())
        # 2) create line graph
        lgG[q] = nx.line_graph(graph_s[q].copy())
        # 3) calculate the angles between two edges from the graph that is now represented by edges in the line graph
        lgG_V[q] = utilsF.lG_edgeVal(lgG[q].copy(),graphD[q],posL[q])
        # 4) run depth first search
        graphTagg[q] = utilsF.dfs_constrained(graph_s[q].copy(),lgG_V[q].copy(),imgBl[q],posL[q],angleA,overlap) 
        
        utilsF.draw_graph_filament_nocolor(imgP[q],graphTagg[q],posL[q],"",'filament')
        plt.savefig(pathsave+'n_graphs/graph{0}.png'.format(q))
    
        plt.close('all')
        no_filaments[q] = len(np.unique(np.asarray(list(graphTagg[q].edges(data='filament')))[:,2]))
        print('filament defined: ',len(np.unique(np.asarray(list(graphTagg[q].edges(data='filament')))[:,2])))
         
    pickle.dump(posL, open(pathsave+'posL.gpickle', 'wb'))
    ###############################################################################
    #
    # data
    # tracking 
    #
    ###############################################################################
    # if already saved this, then load in
    #g_tagged = nx.read_gpickle(pathsave+'tagged_graph.gpickle')
    #graphTagg = g_tagged.copy()
    
    if(len(img_o)<20):
        memKeep = len(img_o)
    else:
        memVal = 20
        memKeep = utilsF.signMem(graphTagg[0:memVal],posL[0:memVal])
    
    # first graph needs unique tags
    for node1, node2, property in graphTagg[0].edges(data=True):
        for n in range(len(graphTagg[0][node1][node2])):
            graphTagg[0][node1][node2][n]['tags'] = property['filament']
        
        
    list(graphTagg[0].edges(data='filament'))
        
    max_tag = np.max(list(graphTagg[0].edges(data='filament')),axis=0)[2] 
    
    g_tagged = [0]*(len(img_o))
    g_tagged[0] = graphTagg[0]
    cost = [0]*(len(img_o)-1)
    tag_new = [0]*(len(img_o))
    tag_new[0] = max_tag
    filamentsNU = []
    
    for i in range(len(img_o)-1):
        g_tagged[i+1],cost[i],tag_new[i+1],filamentsNU = utilsF.filament_tag(g_tagged[i],graphTagg[i+1],posL[i],posL[i+1],tag_new[i],max_cost,filamentsNU,memKeep)
    
    pickle.dump(g_tagged, open(pathsave+'tagged_graph.gpickle', 'wb'))
    
    
    for i in range(len(img_o)):
        title = "graph {}".format(i+1)
        utilsF.draw_graph_filament_track_nocolor(imgP[i],g_tagged[i],posL[i],title,max(tag_new),padv=50)
        pathsave_taggraph = pathsave+"mov/trackgraph{}.png".format(i+1)
        plt.savefig(pathsave_taggraph)
        plt.close('all')
    
    
    ###############################################################################
    #
    # data analysis
    #
    ###############################################################################
    
    plt.rc('xtick', labelsize=24) 
    plt.rc('ytick', labelsize=24) 
        
        
        
    unique_filaments = [0]*len(img_o)
    unique_frames = []
    
    for i in range(len(img_o)):
        unique_filaments[i] = len(np.unique(np.asarray(list(g_tagged[i].edges(data='tags')))[:,2]))
        unique_frames.extend(list(np.unique(np.asarray(list(g_tagged[i].edges(data='tags')))[:,2])))
    
    
    plt.figure(figsize=(10,10))
    plt.scatter(np.arange(0,len(unique_filaments)),unique_filaments)
    plt.xlabel('frames',size=24)
    plt.ylabel('# filaments',size=24)
    plt.savefig(pathsave+'plots/filaments_per_frame.png')
    
    
    ###############################################################################
    #
    # data analysis - one frame at a time
    #
    ###############################################################################
    
    pd_fil_info = utilsF.filament_info_time(imgP, g_tagged, posL, pathsave,imF,maskDraw)
    
    pd_fil_info = pd.read_csv(pathsave+'tracked_filaments_info.csv')

    vals = Counter(pd_fil_info['filament']).values()
    
    counts,bins = np.histogram(list(vals),20)
    plt.figure(figsize=(10,7))
    plt.hist(bins[:-1], bins, weights=counts,color='green')
    plt.xlabel('frames',size=24)
    plt.ylabel('filaments survival',size=24)
    plt.savefig(pathsave+'plots/survival_filaments.png')
    
    counts,bins = np.histogram(list(vals),20,density='True')
    plt.figure(figsize=(10,7))
    plt.hist(bins[:-1], bins, weights=counts,color='green')
    plt.xlabel('frames',size=24)
    plt.ylabel('filaments survival',size=24)
    plt.savefig(pathsave+'plots/survival_filaments_normalized.png')
    
    dens = np.zeros(len(img_o))
    fil_len = np.zeros(len(img_o))
    fil_I = np.zeros(len(img_o))
    for i in range(len(img_o)):
        dens[i] = pd_fil_info[pd_fil_info['frame number']==i]['filament density'].values[0]
        fil_len[i] =np.median(pd_fil_info[pd_fil_info['frame number']==i]['filament length'])
        fil_I[i] = np.median(pd_fil_info[pd_fil_info['frame number']==i]['filament intensity per length'])
        
        
    plt.figure(figsize=(10,10))
    plt.scatter(np.arange(0,len(img_o)),dens)
    plt.xlabel('frames',size=24)
    plt.ylabel('filament density',size=24)
    plt.savefig(pathsave+'plots/filament_density.png')
    
    plt.figure(figsize=(10,10))
    plt.scatter(np.arange(0,len(img_o)),fil_len)
    plt.xlabel('frames',size=24)
    plt.ylabel('filament median length',size=24)
    plt.savefig(pathsave+'plots/filamentlength.png')
    
    
    mean_angle,var_val = utilsF.circ_stat_plot(pathsave,pd_fil_info)
    
    line_mean = np.mean(mean_angle)
    
    plt.figure(figsize=(10,10))
    plt.scatter(np.arange(0,len(mean_angle)),mean_angle)
    plt.plot(np.arange(0,len(mean_angle)),np.ones(len(mean_angle))*line_mean,color='black',linestyle='dashed')
    plt.xlabel('Frames',size=24)
    plt.ylabel('Circular mean angle',size=24)
    plt.savefig(pathsave+'plots/angles_mean.png')
    
    plt.figure(figsize=(10,10))
    plt.scatter(np.arange(0,len(var_val)),var_val)
    plt.xlabel('frames',size=24)
    plt.ylabel('circular variance of angles',size=24)
    plt.savefig(pathsave+'plots/angles_var.png')
    
    tagsU = pd_fil_info['filament'].unique()
    vals = np.zeros(len(tagsU))
    lives = np.zeros(len(tagsU))
    plt.figure(figsize=(10,10))
    for s,m in zip(tagsU,range(len(tagsU))):
        fil = pd_fil_info[pd_fil_info['filament']==s]['filament length'].values
        #print(fil)np.median(fil
        vals[m]=np.median(fil)
        lives[m]=len(fil)
        plt.plot(np.arange(0,len(fil)),fil)
    plt.xlabel('Survival frames',size=24)
    plt.ylabel('filament length',size=24)
    plt.savefig(pathsave+'plots/survival_len.png')
    
    ###############################################################################
    #
    # mean/median value per frame
    #
    ###############################################################################
    
    df_angles2 = pd.DataFrame()
    df_angles2['angles'] = mean_angle
    df_angles2['var'] = var_val
    df_angles2['frame density'] = dens
    df_angles2['filament median length'] = fil_len
    df_angles2['filament mediam intensity per length'] = fil_I
    df_angles2['name'] = name_cell
    
    df_angles2.to_csv(pathsave+'value_per_frame.csv',index=False)
    
    return


def create_all_still(pathsave,img_o,maskDraw,size,eps,thresh_top,sigma,small,angleA,overlap,name_cell):
    # check if folders exists:
    path_check = [pathsave+'n_graphs',pathsave+'circ_stat',pathsave+'mov',pathsave+'plots']
    for i in range(len(path_check)):
        
        if not os.path.exists(path_check[i]):
            os.makedirs(path_check[i])
    
    
    N,P = (img_o.shape)
    imgP=np.zeros((N+2,P+2))
    

    imgP = np.pad(img_o, 1, 'constant')
        
    # 0) create graph
    graph_s, posL, imgSkel, imgAF, imgBl,imF,mask,df_pos = utilsF.creategraph(imgP,size=size,eps=eps,thresh_top=thresh_top,sigma=sigma,small=small)
    #utilsF.draw_graph(imgSkel[q],graph_s[q],posL[q],"untagged graph")
    #plt.savefig(pathsave+'n_graphs/untaggedgraph{0}.png'.format(q))
    # 1) find all dangling edges and mark them
    graphD = utilsF.dangling_edges(graph_s.copy())
    # 2) create line graph
    lgG = nx.line_graph(graph_s.copy())
    # 3) calculate the angles between two edges from the graph that is now represented by edges in the line graph
    lgG_V = utilsF.lG_edgeVal(lgG.copy(),graphD,posL)
    # 4) run depth first search
    graphTagg = utilsF.dfs_constrained(graph_s.copy(),lgG_V.copy(),imgBl,posL,angleA,overlap) 
    
    utilsF.draw_graph_filament_nocolor(imgP,graphTagg,posL,"",'filament')
    plt.savefig(pathsave+'n_graphs/graph.png')

    plt.close('all')
    print('filament defined: ',len(np.unique(np.asarray(list(graphTagg.edges(data='filament')))[:,2])))
    
    
    ###############################################################################
    #
    # data analysis - one frame at a time
    #
    ###############################################################################
    
    pd_fil_info = utilsF.filament_info(imgP, graphTagg, posL, pathsave,imF,maskDraw)
    
    
    pd_fil_info = pd.read_csv(pathsave+'traced_filaments_info.csv')
    
    mean_len = np.mean(pd_fil_info['filament length'])
    
    list_len = np.sort(pd_fil_info['filament length'])
    plt.figure()
    plt.scatter( np.arange(0,len(list_len)),list_len)
    
    
    mean_angle,var_val = utilsF.circ_stat(pd_fil_info,pathsave)
    
    print('mean angle: ', mean_angle, 'circ var: ', var_val, 'mean length: ', mean_len)
    
    return

###############################################################################
#
# load in and run functions
#
###############################################################################

######################
# timeseries

img_o = io.imread("/Users/pgf840/Documents/PLEN/tracking-filaments/dfs/GraFT/tiff/timeseries.tif")
maskDraw = np.ones((img_o.shape[1:3]))
create_all(pathsave = "/Users/pgf840/Documents/PLEN/tracking-filaments/dfs/GraFT/timeseries/",
           img_o=img_o,
           maskDraw=maskDraw,
           size=6,eps=200,thresh_top=0.5,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
           name_cell='in silico time')

######################
# one image

img = io.imread("/Users/pgf840/Documents/PLEN/tracking-filaments/dfs/GraFT/tiff/timeseries.tif")
img_still = img_o[0]
maskDraw = np.ones((img.shape[1:3]))
create_all_still(pathsave = "/Users/pgf840/Documents/PLEN/tracking-filaments/dfs/GraFT/still/",
           img_o=img_still,
           maskDraw=maskDraw,
           size=6,eps=200,thresh_top=0.5,sigma=sigma,small=small,angleA=140,overlap=4,
           name_cell='in silico still')
