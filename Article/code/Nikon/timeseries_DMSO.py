#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:22:14 2023

@author: isabella
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tslearn
#from tslearn.utils import to_time_series_dataset
#from tslearn.clustering import TimeSeriesKMeans, silhouette_score
#from tslearn.barycenters import dtw_barycenter_averaging
#import sys
#from datetime import datetime
import os

plt.close('all')

###############################################################################
#
# create pandas dataset with timeseries format
#
###############################################################################

def df_time(pd_fil_info,pathsave):
    
    path_check = [pathsave+'timeseries']
    for i in range(len(path_check)):
        
        if not os.path.exists(path_check[i]):
            os.makedirs(path_check[i])
    plt.figure(figsize=(10,10))
    
    pd_time = pd.DataFrame()
    tagsU = pd_fil_info['filament'].unique()
    fil_len = []
    fil_tag = []
    fil_angle = []
    fil_intPL = []
    fil_int = []
    fil_frames = []
    mean_len = []
    med_len = []
    mean_angle = []
    med_angle = []
    mean_intPL = []
    med_intPL = []
    mean_int = []
    alive = []
    avg_fil_frames = []
    avg_fil_int = []
    avg_fil_intPL = []
    for s in tagsU:
        time = pd_fil_info[pd_fil_info['filament']==s]['frame number'].values
        fil = pd_fil_info[pd_fil_info['filament']==s]['filament length'].values
        angle = pd_fil_info[pd_fil_info['filament']==s]['filament angle'].values
        intF = pd_fil_info[pd_fil_info['filament']==s]['filament intensity'].values
        intPL = pd_fil_info[pd_fil_info['filament']==s]['filament intensity per length'].values
        if(len(fil)>=5):
            fil_len.append(fil)
            fil_tag.append(s)
            fil_angle.append(angle)
            fil_intPL.append(intPL)
            fil_frames.append(time)
            fil_int.append(intF)
            mean_len.append(np.mean(fil))
            med_len.append(np.median(fil))
            mean_angle.append(np.mean(angle))
            med_angle.append(np.median(angle))
            mean_intPL.append(np.mean(intPL))
            med_intPL.append(np.median(intPL))
            mean_int.append(np.mean(intF))
            alive.append(len(time))
            #plt.plot(time,fil)
            pd_timeIM = pd.DataFrame({'frames':time, 'filament length' : fil,'filament angle': angle, 'intensity per length': intPL,'intensity': intF, 'filament' : s})
            plt.plot(time, pd_timeIM[['filament length']].rolling(5,min_periods=5).mean().values)
            
            pd_timeIM['avg fil length']=pd_timeIM[['filament length']].rolling(5,min_periods=5).mean()
            pd_timeIM['avg fil angle']=pd_timeIM[['filament angle']].rolling(5,min_periods=5).mean()
            pd_timeIM['avg intensity per length']=pd_timeIM[['intensity per length']].rolling(5,min_periods=5).mean()
            pd_timeIM['avg intensity']=pd_timeIM[['intensity']].rolling(5,min_periods=5).mean()
            pd_timeIM = pd_timeIM.dropna()
            
            avg_fil_frames.append(pd_timeIM['avg fil length'].values)
            avg_fil_int.append(pd_timeIM['avg intensity'].values)
            avg_fil_intPL.append(pd_timeIM['avg intensity per length'].values)
            pd_time = pd.concat([pd_time, pd_timeIM], axis=0)
            
    data = {'filament length' : fil_len,'mean length per filament': mean_len,'median length per filament': med_len,
            'filament angle': fil_angle,'mean filament angle per filament':mean_angle, 'median filament angle per filament': med_angle,
            'intensity per filament':fil_int,'mean intensity per filament': mean_int,
            'intensity per length': fil_intPL,'mean intensity per length':mean_intPL,'median intensity per length': med_intPL,
            'frames':fil_frames,'smooth5 filament length': avg_fil_frames,
            'smooth5 filament intensity':avg_fil_int,'smooth5 filament intensity per length':avg_fil_intPL,'filament' : fil_tag}
    frame = pd.DataFrame.from_dict(data, orient='index')
    frame = frame.transpose()
    
            
    plt.xlabel('frames')
    plt.ylabel('length')      
    plt.savefig(pathsave+'/timeseries/mean_length.png')

    return(frame)
        

def df_time_move(trackPD,pathsave): 
    tagsU = trackPD['filament tag'].unique()
    fil_frames = []
    move = []
    fil_tag = []

    for s in tagsU:
        time = trackPD[trackPD['filament tag']==s]['frame'].values
        fil_move = trackPD[trackPD['filament tag']==s]['mean move'].values
        if(len(time)>=5):
            fil_tag.append(s)
            move.append(fil_move)
            fil_frames.append(time)
            
            pd_timeIM = pd.DataFrame({'fil move':fil_move})
            plt.plot(time, pd_timeIM[['fil move']].rolling(5,min_periods=5).mean().values)

            
            
    data = {'filament movement' : move,
            'frames':fil_frames,'filament' : fil_tag}
    frame = pd.DataFrame.from_dict(data, orient='index')
    frame = frame.transpose()
    
            
    plt.xlabel('frames')
    plt.ylabel('length')      
    plt.savefig(pathsave+'/timeseries/mean_move.png')

    return(frame)

###############################################################################
#
# day load in
#
###############################################################################


pathsave_DMSO = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO/"
pathsave_DSF = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DSF/"
pathsave_flg22 = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/flg22/"

tm = ['cell1/','cell2/','cell3/','cell4/','cell5/']

pathsave_DMSO_all = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/'
#control
pd_fil_info11 = pd.read_csv(pathsave_DMSO+tm[0]+'tracked_filaments_info.csv')
pd_fil_info12 = pd.read_csv(pathsave_DMSO+tm[1]+'tracked_filaments_info.csv')
pd_fil_info13 = pd.read_csv(pathsave_DMSO+tm[2]+'tracked_filaments_info.csv')
pd_fil_info14 = pd.read_csv(pathsave_DMSO+tm[3]+'tracked_filaments_info.csv')
#DSF
pd_fil_info21 = pd.read_csv(pathsave_DSF+tm[1]+'tracked_filaments_info.csv')
pd_fil_info22 = pd.read_csv(pathsave_DSF+tm[2]+'tracked_filaments_info.csv')
pd_fil_info23 = pd.read_csv(pathsave_DSF+tm[3]+'tracked_filaments_info.csv')
pd_fil_info24 = pd.read_csv(pathsave_DSF+tm[4]+'tracked_filaments_info.csv')
#flg22
pd_fil_info31 = pd.read_csv(pathsave_flg22+tm[0]+'tracked_filaments_info.csv')
pd_fil_info32 = pd.read_csv(pathsave_flg22+tm[1]+'tracked_filaments_info.csv')
pd_fil_info33 = pd.read_csv(pathsave_flg22+tm[2]+'tracked_filaments_info.csv')
pd_fil_info34 = pd.read_csv(pathsave_flg22+tm[4]+'tracked_filaments_info.csv')

#p1
frame11 = df_time(pd_fil_info11,pathsave_DMSO)
frame12 = df_time(pd_fil_info12,pathsave_DMSO)
frame13 = df_time(pd_fil_info13,pathsave_DMSO)
frame14 = df_time(pd_fil_info14,pathsave_DMSO)
#p2
frame21 = df_time(pd_fil_info21,pathsave_DSF)
frame22 = df_time(pd_fil_info22,pathsave_DSF)
frame23 = df_time(pd_fil_info23,pathsave_DSF)
frame24 = df_time(pd_fil_info24,pathsave_DSF)
#p3
frame31 = df_time(pd_fil_info31,pathsave_flg22)
frame32 = df_time(pd_fil_info32,pathsave_flg22)
frame33 = df_time(pd_fil_info33,pathsave_flg22)
frame34 = df_time(pd_fil_info34,pathsave_flg22)


plt.close('all')

frames_control = pd.concat([frame11, frame12, frame13, frame14], ignore_index=True)
frames_10 = pd.concat([frame21, frame22, frame23, frame24], ignore_index=True)
frames_20 = pd.concat([frame31, frame32, frame33, frame34], ignore_index=True)

frames_control.to_csv(pathsave_DMSO_all+'frames_DMSOcontrol.csv')
frames_10.to_csv(pathsave_DMSO_all+'frames_DSF.csv')
frames_20.to_csv(pathsave_DMSO_all+'frames_flg22.csv')
