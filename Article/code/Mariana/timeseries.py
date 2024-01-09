#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:07:38 2023

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from datetime import datetime
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


pathsave = "/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/"

plant=['p1','p2','p3','p4','p6','p7','p8']
tm = ['_top/','_mid/','_bot/']

pathsave_day = '/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/'
#p1
pd_fil_info11 = pd.read_csv(pathsave+plant[0]+tm[0]+'tracked_filaments_info.csv')
pd_fil_info12 = pd.read_csv(pathsave+plant[0]+tm[1]+'tracked_filaments_info.csv')
pd_fil_info13 = pd.read_csv(pathsave+plant[0]+tm[2]+'tracked_filaments_info.csv')
#p2
pd_fil_info21 = pd.read_csv(pathsave+plant[1]+tm[0]+'tracked_filaments_info.csv')
pd_fil_info22 = pd.read_csv(pathsave+plant[1]+tm[1]+'tracked_filaments_info.csv')
pd_fil_info23 = pd.read_csv(pathsave+plant[1]+tm[2]+'tracked_filaments_info.csv')
#p3
pd_fil_info31 = pd.read_csv(pathsave+plant[2]+tm[0]+'tracked_filaments_info.csv')
pd_fil_info32 = pd.read_csv(pathsave+plant[2]+tm[1]+'tracked_filaments_info.csv')
pd_fil_info33 = pd.read_csv(pathsave+plant[2]+tm[2]+'tracked_filaments_info.csv')
#p4
pd_fil_info43 = pd.read_csv(pathsave+plant[3]+tm[2]+'tracked_filaments_info.csv')
#p6
pd_fil_info51 = pd.read_csv(pathsave+plant[4]+tm[0]+'tracked_filaments_info.csv')
pd_fil_info52 = pd.read_csv(pathsave+plant[4]+tm[1]+'tracked_filaments_info.csv')
#p7
pd_fil_info61 = pd.read_csv(pathsave+plant[5]+tm[0]+'tracked_filaments_info.csv')
pd_fil_info62 = pd.read_csv(pathsave+plant[5]+tm[1]+'tracked_filaments_info.csv')
pd_fil_info63 = pd.read_csv(pathsave+plant[5]+tm[2]+'tracked_filaments_info.csv')
#p8
pd_fil_info71 = pd.read_csv(pathsave+plant[6]+tm[0]+'tracked_filaments_info.csv')
pd_fil_info72 = pd.read_csv(pathsave+plant[6]+tm[1]+'tracked_filaments_info.csv')
pd_fil_info73 = pd.read_csv(pathsave+plant[6]+tm[2]+'tracked_filaments_info.csv')

#p1
frame11 = df_time(pd_fil_info11,pathsave)
frame12 = df_time(pd_fil_info12,pathsave)
frame13 = df_time(pd_fil_info13,pathsave)
#p2
frame21 = df_time(pd_fil_info21,pathsave)
frame22 = df_time(pd_fil_info22,pathsave)
frame23 = df_time(pd_fil_info23,pathsave)
#p3
frame31 = df_time(pd_fil_info31,pathsave)
frame32 = df_time(pd_fil_info32,pathsave)
frame33 = df_time(pd_fil_info33,pathsave)
#p4
frame43 = df_time(pd_fil_info43,pathsave)
#p6
frame51 = df_time(pd_fil_info51,pathsave)
frame52 = df_time(pd_fil_info52,pathsave)
#p7
frame61 = df_time(pd_fil_info61,pathsave)
frame62 = df_time(pd_fil_info62,pathsave)
frame63 = df_time(pd_fil_info63,pathsave)
#p8
frame71 = df_time(pd_fil_info71,pathsave)
frame72 = df_time(pd_fil_info72,pathsave)
frame73 = df_time(pd_fil_info73,pathsave)

plt.close('all')

frames_alltop = pd.concat([frame11, frame21, frame31, frame51, frame61, frame71], ignore_index=True)
frames_allmid = pd.concat([frame12, frame22, frame32, frame52, frame62, frame72], ignore_index=True)
frames_allbot = pd.concat([frame13, frame23, frame33, frame43, frame63, frame73], ignore_index=True)

frames_alltop.to_csv(pathsave_day+'frames_top.csv')
frames_allmid.to_csv(pathsave_day+'frames_mid.csv')
frames_allbot.to_csv(pathsave_day+'frames_bot.csv')

###############################################################################
#
# creation
#
###############################################################################


'''
import tslearn
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.barycenters import dtw_barycenter_averaging
###################
# length (no smoothing) with 5
#top

Xalltop = to_time_series_dataset(frames_alltop['filament length'].values)
print(datetime.now())
dist_Xtop = tslearn.metrics.cdist_dtw(Xalltop)
print(datetime.now())
print('X top cdistance mean: ',np.mean(dist_Xtop))
np.save(pathsave_day+'top_fil_len.npy',dist_Xtop)


#bottom

Xallbottom = to_time_series_dataset(frames_allbot['filament length'].values)
print(datetime.now())
dist_Xbottom = tslearn.metrics.cdist_dtw(Xallbottom)
print(datetime.now())
print('X ylatb cdistance mean: ',np.mean(dist_Xbottom))
np.save(pathsave_day+'bot_fil_len.npy',dist_Xbottom)

#middle

Xallmiddle = to_time_series_dataset(frames_allmid['filament length'].values)
print(datetime.now())
dist_Xmiddle = tslearn.metrics.cdist_dtw(Xallmiddle)
print(datetime.now())
print('X ylatb cdistance mean: ',np.mean(dist_Xmiddle))
np.save(pathsave_day+'middle_fil_len.npy',dist_Xmiddle)


###################
# filament angle (no smoothing) with 5

# top
Xangletop = to_time_series_dataset(frames_alltop['filament angle'].values)
print(datetime.now())
angle_Xtop = tslearn.metrics.cdist_dtw(Xangletop)
print(datetime.now())
print('X ylatb cdistance angle: ',np.mean(angle_Xtop))
np.save(pathsave_day+'top_fil_angle.npy',angle_Xtop)

# bottom
Xanglebottom = to_time_series_dataset(frames_allbot['filament angle'].values)
print(datetime.now())
angle_Xbottom = tslearn.metrics.cdist_dtw(Xanglebottom)
print(datetime.now())
print('X ylatb cdistance angle: ',np.mean(angle_Xbottom))
np.save(pathsave_day+'bot_fil_angle.npy',angle_Xbottom)

# middle
Xanglemiddle = to_time_series_dataset(frames_allmid['filament angle'].values)
print(datetime.now())
angle_Xmiddle = tslearn.metrics.cdist_dtw(Xanglemiddle)
print(datetime.now())
print('X ylatb cdistance angle: ',np.mean(angle_Xmiddle))
np.save(pathsave_day+'middle_fil_angle.npy',angle_Xmiddle)


###################
# intensity per length
# bottom
Xintlentop = to_time_series_dataset(frames_alltop['intensity per length'].values)
print(datetime.now())
intlen_Xtop = tslearn.metrics.cdist_dtw(Xintlentop)
print(datetime.now())
print('X bottom cdistance intensity per length: ',np.mean(intlen_Xtop))
np.save(pathsave_day+'top_fil_intPL.npy',intlen_Xtop)

# bottom
Xintlenbottom = to_time_series_dataset(frames_allbot['intensity per length'].values)
print(datetime.now())
intlen_Xbottom = tslearn.metrics.cdist_dtw(Xintlenbottom)
print(datetime.now())
print('X bottom cdistance intensity per length: ',np.mean(intlen_Xbottom))
np.save(pathsave_day+'bot_fil_intPL.npy',intlen_Xbottom)

# middle
Xintlemiddle = to_time_series_dataset(frames_allmid['intensity per length'].values)
print(datetime.now())
intlen_Xmiddle = tslearn.metrics.cdist_dtw(Xintlemiddle)
print(datetime.now())
print('X middle cdistance intensity per length: ',np.mean(intlen_Xmiddle))
np.save(pathsave_day+'middle_fil_intPL.npy',intlen_Xmiddle)



###################
# intensity per filament
#top
X_intfilbottom = to_time_series_dataset(frames_alltop['intensity per filament'].values)
print(datetime.now())
intfil_Xbottom = tslearn.metrics.cdist_dtw(X_intfilbottom)
print(datetime.now())
print('X bottom cdistance intensity per filament: ',np.mean(intfil_Xbottom))
np.save(pathsave_day+'top_fil_intfil.npy',intfil_Xbottom)

#bottom
X_intfilbottom = to_time_series_dataset(frames_allbot['intensity per filament'].values)
print(datetime.now())
intfil_Xbottom = tslearn.metrics.cdist_dtw(X_intfilbottom)
print(datetime.now())
print('X bottom cdistance intensity per filament: ',np.mean(intfil_Xbottom))
np.save(pathsave_day+'bot_fil_intfil.npy',intfil_Xbottom)

#middle
X_intfilmiddle = to_time_series_dataset(frames_allmid['intensity per filament'].values)
print(datetime.now())
intfil_Xmiddle = tslearn.metrics.cdist_dtw(X_intfilmiddle)
print(datetime.now())
print('X middle cdistance intensity per filament: ',np.mean(intfil_Xmiddle))
np.save(pathsave_day+'middle_fil_intfil.npy',intfil_Xmiddle)

###################
# change in angle

##### change in angle per filament
def change_angles(data):
    l_mAL = []

    for l in range(len(data)):
        list_angle = data[l]
        mAL = np.zeros(len(list_angle)-1)
        for i in range(len(list_angle)-1):
            mAL[i] = list_angle[i]-list_angle[i+1]
        l_mAL.append(mAL)
    return l_mAL

l_mAL_top = change_angles(frames_alltop['filament angle'])
l_mAL_bottom = change_angles(frames_allbot['filament angle'])
l_mAL_biddle = change_angles(frames_allmid['filament angle'])

X_changeb = to_time_series_dataset(l_mAL_bottom)
print(datetime.now())
changeA_Xbottom = tslearn.metrics.cdist_dtw(X_changeb)
print(datetime.now())
print('X bottom change angle per filament: ',np.mean(changeA_Xbottom))
np.save(pathsave_day+'top_fil_changeangle.npy',changeA_Xbottom)


l_mAL_middle = change_angles(frames_allmid['filament angle'])

X_changeAm = to_time_series_dataset(l_mAL_middle)
print(datetime.now())
changeA_Xmiddle = tslearn.metrics.cdist_dtw(X_changeAm)
print(datetime.now())
print('X middle change angle per filament: ',np.mean(changeA_Xmiddle))
np.save(pathsave_day+'middle_fil_changeangle.npy',changeA_Xmiddle)

###############################################################################
# mean movement

def creation_of_data_ml(list_vals):
    fullTrack=pd.DataFrame()
    for i in range(len(list_vals)):
        trackCur = pd.read_csv(list_vals[i]+'tracked_move.csv')
        tracklen = pd.read_csv(list_vals[i]+'tracked_filaments_info.csv')
        tagsU = tracklen['filament'].unique()
        move_pL = []
        fil_tag = []
        mean_move = []
        for s in tagsU:
            fil = tracklen[tracklen['filament']==s]['filament length'].values
            move = trackCur[trackCur['filament tag']==s]['mean move'].values
            if(len(fil)>=5):
                move_pL.append(move/fil[0:len(move)])
                mean_move.append(move)
                fil_tag.append(s)
                
        data = {'filament movement per length' : move_pL,'filament movement': mean_move, 'filament' : fil_tag}
        frame = pd.DataFrame.from_dict(data, orient='index')
        frame = frame.transpose()
        
        fullTrack = pd.concat(([fullTrack,frame]),ignore_index=True)
    return fullTrack

##########
# top
list_vals=[plant[0]+tm[0], plant[1]+tm[0], plant[2]+tm[0], plant[4]+tm[0], plant[5]+tm[0], plant[6]+tm[0]]
df_top = creation_of_data_ml(list_vals)

# mean movement per length
X_movelbottom = to_time_series_dataset(df_top['filament movement per length'].values)
print(datetime.now())
mmove_Xbottom = tslearn.metrics.cdist_dtw(X_movelbottom)
print(datetime.now())
print('X top cdistance mean move: ',np.mean(mmove_Xbottom))
np.save(pathsave_day+'top_fil_meanmovel.npy',mmove_Xbottom)

##########
# bottom
list_vals=[plant[0]+tm[2], plant[1]+tm[2], plant[2]+tm[2], plant[3]+tm[2], plant[5]+tm[2], plant[6]+tm[2]]
df_bottom = creation_of_data_ml(list_vals)

# mean movement per length
X_movelbottom = to_time_series_dataset(df_bottom['filament movement per length'].values)
print(datetime.now())
mmove_Xbottom = tslearn.metrics.cdist_dtw(X_movelbottom)
print(datetime.now())
print('X bottom cdistance mean move: ',np.mean(mmove_Xbottom))
np.save(pathsave_day+'bottom_fil_meanmovel.npy',mmove_Xbottom)

##########
# middle
list_vals=[plant[0]+tm[1], plant[1]+tm[1], plant[2]+tm[1], plant[4]+tm[1], plant[5]+tm[1], plant[6]+tm[1]]
df_middle= creation_of_data_ml(list_vals)

# mean movement
X_movelmiddle= to_time_series_dataset(df_middle['filament movement per length'].values)
print(datetime.now())
mmove_Xmiddle = tslearn.metrics.cdist_dtw(X_movelmiddle)
print(datetime.now())
print('X middle cdistance mean move: ',np.mean(mmove_Xmiddle))
np.save(pathsave_day+'middle_fil_meanmovel.npy',mmove_Xmiddle)


###################
# mean movement top
X_movebottom = to_time_series_dataset(df_top['filament movement'].values)
print(datetime.now())
mmove_bottom = tslearn.metrics.cdist_dtw(X_movebottom)
print(datetime.now())
print('X top cdistance mean move: ',np.mean(mmove_bottom))

np.save(pathsave_day+'top_fil_meanmove.npy',mmove_bottom)

###################
# mean movement bottom
X_movebottom = to_time_series_dataset(df_bottom['filament movement'].values)
print(datetime.now())
mmove_bottom = tslearn.metrics.cdist_dtw(X_movebottom)
print(datetime.now())
print('X bottom cdistance mean move: ',np.mean(mmove_bottom))

np.save(pathsave_day+'bottom_fil_meanmove.npy',mmove_bottom)

###################
# mean movement middle
X_movemiddle = to_time_series_dataset(df_middle['filament movement'].values)
print(datetime.now())
mmove_middle = tslearn.metrics.cdist_dtw(X_movemiddle)
print(datetime.now())
print('X middle cdistance mean move: ',np.mean(mmove_middle))

np.save(pathsave_day+'middle_fil_meanmove.npy',mmove_middle)
'''