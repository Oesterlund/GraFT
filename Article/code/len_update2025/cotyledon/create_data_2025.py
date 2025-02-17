#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:20:28 2025

@author: isabella
"""

import numpy as np
import pandas as pd
import re
import astropy.stats
import plotly.graph_objects as go
from astropy import units as u

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/cotyledon/figs/'

pixelS = 0.217

###############################################################################
#
# functions
#
###############################################################################

def creation_of_data_ml(list_vals):
    fullTrack=pd.DataFrame()
    for i in range(len(list_vals)):
        trackCur = pd.read_csv(list_vals[i]+'tracked_move.csv')
        tracklen = pd.read_csv(list_vals[i]+'tracked_filaments_info.csv')
        tagsU = tracklen['filament'].unique()
        move_pL = []
        fil_tag = []
        mean_move = []
        mean_len = []
        mean_angle = []
        filBend=[]
        filInt = []
        filtIntLen=[]
        for s in tagsU:
            fil = tracklen[tracklen['filament']==s]['filament length'].values
            move = trackCur[trackCur['filament tag']==s]['mean move'].values
            angle = tracklen[tracklen['filament']==s]['filament angle'].values
            filB = tracklen[tracklen['filament']==s]['filament bendiness'].values
            filI = tracklen[tracklen['filament']==s]['filament intensity'].values
            filIl = tracklen[tracklen['filament']==s]['filament intensity per length'].values
            if(len(fil)>=5):
                move_pL.append(np.mean(move)/np.mean(fil))
                mean_len.append(np.mean(fil))
                mean_move.append(np.mean(move))
                mean_angle.append(np.mean(angle))
                fil_tag.append(s)
                filB[filB>1]=1
                filBend.append(np.mean(filB))
                filInt.append(np.mean(filI))
                filtIntLen.append(np.mean(filIl))
                
        data = {'filament movement per length' : move_pL,'mean filament movement': mean_move, 'filament' : fil_tag, 'mean filament length': mean_len,
                'mean filament angle': mean_angle,'mean filament bendiness':filBend, 'mean intensity': filInt,'mean intensity per length':filtIntLen}
        frame = pd.DataFrame.from_dict(data)
        
        fullTrack = pd.concat(([fullTrack,frame]),ignore_index=True)
    return fullTrack

def creation_of_dens(list_vals):
    fullTrack=pd.DataFrame()
    for i in range(len(list_vals)):
        tracklen = pd.read_csv(list_vals[i]+'tracked_filaments_info.csv')
        tagsU = tracklen['frame number'].unique()
        mean_dens=[]
        for s in tagsU:
            fdens = tracklen[tracklen['frame number']==s]['filament density'].values
        mean_dens.append(np.mean(fdens))

        data = {'mean density': mean_dens}
        frame = pd.DataFrame.from_dict(data)
        
        fullTrack = pd.concat(([fullTrack,frame]),ignore_index=True)
    return fullTrack

###############################################################################
#
# import files
#
###############################################################################


df_Dtop = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/frames_top.csv')
df_Dtop['cell name'] = 'Young'
df_Dmid = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/frames_mid.csv')
df_Dmid['cell name'] = 'Expanding'
df_Dbot = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/frames_bot.csv')
df_Dbot['cell name'] = 'Mature'

df_all = pd.concat([df_Dtop,df_Dmid,df_Dbot], axis=0,ignore_index=True) 

##### change in angle per filament
l_mAL = []
std_l_mAL = np.zeros(len(df_all))
mean_l_mAL = np.zeros(len(df_all))
median_l_mAL = np.zeros(len(df_all))
mean_l_mlL = np.zeros(len(df_all))
median_l_mlL = np.zeros(len(df_all))
for l in range(len(df_all)):
    list_angle = [float(s) for s in re.findall(r'[-+]?(?:\d*\.*\d+)', df_all['filament angle'][l])]
    mAL = np.zeros(len(list_angle)-1)
    list_length = [float(s) for s in re.findall(r'[-+]?(?:\d*\.*\d+)', df_all['filament length'][l])]
    mlL = np.zeros(len(list_length)-1)
    for i in range(len(list_angle)-1):
        mAL[i] = list_angle[i]-list_angle[i+1]
        
    for m in range(len(list_length)-1):
        mlL[m] = list_length[m]-list_length[m+1]
        
    std_l_mAL[l] = np.std(mAL)
    mean_l_mAL[l] = np.mean(mAL)
    median_l_mAL[l] = np.median(mAL)
    l_mAL.append(mAL)
    mean_l_mlL[l] = np.mean(mlL)
    median_l_mlL[l] = np.median(mlL)

df_all['mean angle change per filament']=mean_l_mAL
df_all['median angle change per filament']=median_l_mAL
df_all['std of mean angle change per filament'] = std_l_mAL

df_all['mean length change per filament']=mean_l_mlL
df_all['median length change per filament']=median_l_mlL
        
#################################
# creation into 4 groups for len change 

cutval1 = 3.2164406150564564/pixelS
cutval2 = 6.148265832210858/pixelS
cutval3 = 16.262615814598842/pixelS
df_all['cell name 2'] = df_all['cell name'] 
df_all['cell name 3'] = df_all['cell name'] 
cell_name_list = ['Young', 'Expanding','Mature']
df_all['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
    #print(i,m)
    df_all['sort']=0
    df_all['cell name 2'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i)) ] = '{0} <3.22'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i)) ] = '<3.22'
    df_all['sort2'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i))] = 1 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = '{0} 3.22-6.15'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = '3.22-6.15'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = 2 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = '{0} 6.15-16.26'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = '6.15-16.26'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = 3 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = '{0} >16.26'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = '>16.26'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = 4 + 4*m
    
    print(i)
    print(len(df_all[(df_all['sort2']==1+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==1+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==2+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==2+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==3+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==3+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))

for i,m in zip(cell_name_list,np.arange(0,4)):

    print(i)
    print(len(df_all[(df_all['sort2']==1+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==1+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==2+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==2+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==3+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==3+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print('mean very long group: ',np.mean(df_all['mean length per filament'][(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)]*pixelS))

df_all=df_all.sort_values(by=['sort2'])

df_all.to_csv(pathsave+'df_all.csv')
###############################################################################
#
# division into 4 groups for creation of data ml
#
###############################################################################

plant=['p1','p2','p3','p4','p6','p7','p8']
tm = ['_top/','_mid/','_bot/']
pathsave2 = '/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/'

list_vals=[pathsave2+plant[0]+tm[0], pathsave2+plant[1]+tm[0], pathsave2+plant[2]+tm[0], pathsave2+plant[4]+tm[0], pathsave2+plant[5]+tm[0], pathsave2+plant[6]+tm[0]]
df_top= creation_of_data_ml(list_vals)
dens_top=creation_of_dens(list_vals)
dens_top['cell name'] = 'Young'
np.mean(dens_top['mean density'])
# middle
list_vals=[pathsave2+plant[0]+tm[1],pathsave2+ plant[1]+tm[1], pathsave2+plant[2]+tm[1], pathsave2+plant[4]+tm[1], pathsave2+plant[5]+tm[1], pathsave2+plant[6]+tm[1]]
df_middle= creation_of_data_ml(list_vals)
dens_middle=creation_of_dens(list_vals)
dens_middle['cell name'] = 'Expanding'
np.mean(dens_middle['mean density'])
# bottom
list_vals=[pathsave2+plant[0]+tm[2],pathsave2+ plant[1]+tm[2], pathsave2+plant[2]+tm[2], pathsave2+plant[3]+tm[2], pathsave2+plant[5]+tm[2], pathsave2+plant[6]+tm[2]]
df_bottom= creation_of_data_ml(list_vals)
dens_bottom=creation_of_dens(list_vals)
dens_bottom['cell name'] = 'Mature'
np.mean(dens_bottom['mean density'])

df_dens = pd.concat([dens_top,dens_middle, dens_bottom], axis=0,ignore_index=True) 

df_daytop = pd.concat([df_top], ignore_index=True)
df_daytop['cell name'] = 'Young'

df_daymid = pd.concat([df_middle], ignore_index=True)
df_daymid['cell name'] = 'Expanding'

df_daybot = pd.concat([df_bottom], ignore_index=True)
df_daybot['cell name'] = 'Mature'

df_allFullm = pd.concat([df_daytop,df_daymid, df_daybot], axis=0,ignore_index=True) 

#cutval1 = 50
#cutval2 = 100
#cutval3 = 300
df_allFullm['cell name 2'] = df_allFullm['cell name'] 
df_allFullm['cell name 3'] = df_allFullm['cell name'] 
cell_name_list = ['Young', 'Expanding','Mature']
df_allFullm['sort']=0
df_allFullm['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
    print(i,m)
    df_allFullm['sort'].loc[(df_allFullm['cell name'] == i)]=m
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i)) ] = '{0} <3.22'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i)) ] = '<3.22'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i))] = 1 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = '{0} 3.22-6.15'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = '3.22-6.15'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = 2 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = '{0} 6.15-16.26'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = '6.15-16.26'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = 3 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = '{0} >16.26'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = '>16.26'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = 4 + 4*m

df_allFullm=df_allFullm.sort_values(by=['sort2'])
df_allFullm['mean filament movement']=df_allFullm['mean filament movement']*0.217
df_allFullm['mean intensity per length']=df_allFullm['mean intensity per length']/0.217
df_allFullm['mean filament length']=df_allFullm['mean filament length']*0.217


df_allFullm.to_csv(pathsave+'df_allFullm.csv')

###############################################################################
#
# create circular data
#
###############################################################################

def circ_stat(DataSet):
    
    data = DataSet["mean filament angle"]*np.pi/180. # has to be in radians for astropy to work *u.deg
    weight = DataSet['mean filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))*180/np.pi # switch back to angles
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))
    return mean_angle,var_val

np.unique(df_allFullm['cell name 2'])

up_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young <3.22'])
up_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young 3.22-6.15'])
up_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young 6.15-16.26'])
up_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young >16.26'])

mid_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding <3.22'])
mid_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding 3.22-6.15'])
mid_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding 6.15-16.26'])
mid_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding >16.26'])

bot_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature <3.22'])
bot_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature 3.22-6.15'])
bot_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature 6.15-16.26'])
bot_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature >16.26'])


list_up = [up_angle50[0],up_angle100[0],up_angle300[0],up_angleM300[0]]
list_mid = [mid_angle50[0],mid_angle100[0],mid_angle300[0],mid_angleM300[0]]
list_bot = [bot_angle50[0],bot_angle100[0],bot_angle300[0],bot_angleM300[0]]

list_up1 = [up_angle50[1],up_angle100[1],up_angle300[1],up_angleM300[1]]
list_mid1 = [mid_angle50[1],mid_angle100[1],mid_angle300[1],mid_angleM300[1]]
list_bot1 = [bot_angle50[1],bot_angle100[1],bot_angle300[1],bot_angleM300[1]]

files = [list_up,list_mid,list_bot]
filesV = np.asarray([list_up1,list_mid1,list_bot1])
namesD = ['<3.22', '3.22-6.15', '6.15-16.26', '>16.26']
name_cell = ['Young','Expanding','Mature']

angleFil=pd.DataFrame()
for i in range(len(name_cell)):
    dataang = {'angle' : files[i],'circular variance':filesV[i],'name': namesD,'cells': name_cell[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)
    
angleFil.to_csv(pathsave+'df_angles.csv')
