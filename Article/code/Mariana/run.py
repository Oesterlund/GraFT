#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:50:36 2023

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#import matplotlib.cm as cmx
import os
#import skimage
import skimage.io as io
import networkx as nx
import pandas as pd
#from skimage import util 
#from skimage.transform import rescale
#from skimage.color import rgb2gray,rgba2rgb
#from skimage.draw import (line, bezier_curve, circle_perimeter)
#from scipy.optimize import linear_sum_assignment
#import imageio
#from scipy import ndimage
from collections import Counter
import scipy.io
from scipy import stats
import astropy.stats
import plotly.graph_objects as go
from astropy import units as u
import pickle
import sys

path="/home/isabella/Documents/PLEN/dfs/data/time_dark_seedlings/"
sys.path.append(path)

import utilsF

plt.close('all')

sigma = 1.0                                                                       # tubeness filter width
block = 1001.0                                                                     # adaptive median filter block size
small = 25.0          

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
    
    #graphTagg = g_tagged.copy()
    
    memKeep = utilsF.signMem(graphTagg[0:20],posL[0:20])
    
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
    
    # if already saved this, then load in
    #g_tagged = nx.´(pathsave+'tagged_graph.gpickle')
    #file = open(pathsave+'posL.gpickle','rb')
    #posL = pickle.load(file)
    #file.close()

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
    
    fullTrack = utilsF.track_move(g_tagged,posL,img_o,memKeep, max_cost,pathsave,pd_fil_info)
    
    tagsM = np.unique(fullTrack['filament tag'])
    listMM = np.zeros(len(tagsM))
    listLife = np.zeros(len(tagsM))
    listLength = np.zeros(len(tagsM))
    for i,n in zip(tagsM,range(len(tagsM))):
        listMM[n] = np.mean(fullTrack['mean move'][fullTrack['filament tag']==i])
        listLife[n] = len(fullTrack['mean move'][fullTrack['filament tag']==i])
        listLength[n] = np.mean(pd_fil_info['filament length'][pd_fil_info['filament']==i])
        
    plt.figure(figsize=(10,7))
    plt.scatter(listLife,listMM,color='orange')
    plt.xlabel('Filament survival',size=24)
    plt.ylabel('movement',size=24)
    plt.savefig(pathsave+'plots/movement_filaments.png')
    
    plt.figure(figsize=(10,7))
    plt.scatter(listLength,listMM)
    plt.xlabel('Filament length',size=24)
    plt.ylabel('movement',size=24)
    plt.savefig(pathsave+'plots/movement_filaments_length.png')
    
    key = Counter(pd_fil_info['filament']).keys()
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
    
    
    #mean_angle,var_val = utils4.circ_stat_plot(pathsave,pd_fil_info)
    
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
    
    counts,bins = np.histogram(list(lives),20)
    plt.figure()
    plt.hist(bins[:-1], bins, weights=counts,color='green')
    
        
    plt.figure(figsize=(10,10))
    plt.scatter(lives,vals)
     
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

###############################################################################
#
# load in and run functions
#
###############################################################################

###############################################################################
#
#  plant 1
#
###############################################################################

###################################
# timeseries

'''
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p1/p1_50p_600ms_200f_1s_top2-1.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p1/maskTop.tif')
plt.figure()
plt.imshow(img[0]*maskDraw,cmap='gray_r')
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p1_top/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p1_top')

# middle
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p1/p1_50p_600ms_200f_1s_mid.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p1/maskMid.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p1_mid/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p1_mid')


# bot
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p1/p1_50p_600ms_200f_1s_bot.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p1/maskBot.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p1_bot/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p1_bot')


###############################################################################
#
#  plant 2
#
###############################################################################

img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p2/p2_50p_600ms_200f_1s_top.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p2/maskTop.tif')
plt.figure()
plt.imshow(img[0]*maskDraw,cmap='gray_r')
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p2_top/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p2_top')

# middle
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p2/p2_50p_600ms_200f_1s_mid.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p2/maskMid.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p2_mid/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p2_mid')


# bot
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p2/p2_50p_600ms_200f_1s_bot.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p2/maskBot.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p2_bot/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p2_bot')

###############################################################################
#
#  plant 3
#
###############################################################################

img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p3/p3_50p_600ms_200f_1s_top.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p3/maskTop.tif')
plt.figure()
plt.imshow(img[0]*maskDraw,cmap='gray_r')
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p3_top/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p3_top')

# middle
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p3/p3_50p_600ms_200f_1s_mid.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p3/maskMid.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p3_mid/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p3_mid')


# bot
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p3/p3_50p_600ms_200f_1s_bot.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p3/maskBot.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p3_bot/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p3_bot')

###############################################################################
#
#  plant 4
#
###############################################################################

# bot
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p4/p4_50p_600ms_200f_1s_bot.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p4/maskBot.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p4_bot/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.2,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p4_bot')

###############################################################################
#
#  plant 6
#
###############################################################################


img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p6/p6_50p_600ms_200f_1s_top.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p6/maskTop.tif')
plt.figure()
plt.imshow(img[0]*maskDraw,cmap='gray_r')
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p6_top/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p6_top')

# middle
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p6/p6_50p_600ms_200f_1s_mid.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p6/maskMid.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p6_mid/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p6_mid')


###############################################################################
#
#  plant 7
#
###############################################################################

img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p7/p7_50p_600ms_200f_1s_top.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p7/maskTop.tif')
plt.figure()
plt.imshow(img[0]*maskDraw,cmap='gray_r')
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p7_top/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p7_top')

# middle
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p7/p7_50p_600ms_200f_1s_mid-1.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p7/maskMid.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p7_mid/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p7_mid')

# bot
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p7/p7_50p_600ms_200f_1s_bot.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p7/maskBot.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p7_bot/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p7_bot')


###############################################################################
#
#  plant 8
#
###############################################################################

img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p8/p8_50p_600ms_200f_1s_top2.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p8/maskTop.tif')
plt.figure()
plt.imshow(img[0]*maskDraw,cmap='gray_r')
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p8_top/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p8_top')

# middle
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p8/p8_50p_600ms_200f_1s_mid.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p8/maskMid.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p8_mid/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p8_mid')

'''
# bot
img = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p8/p8_50p_600ms_200f_1s_bot.tif')
maskDraw = io.imread('/home/isabella/Documents/PLEN/dfs/data/seedlings/2023-04-08/p8/maskBot.tif')
plt.figure()
plt.imshow(img[0]*maskDraw)
img_o = img[0:100]*maskDraw
create_all(pathsave= "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/p8_bot/",
                        img_o=img_o,
                        maskDraw=maskDraw,
                        size=4,eps=100,thresh_top=0.1,sigma=sigma,small=small,angleA=140,overlap=4,max_cost=100,
                        name_cell='p8_bot')
