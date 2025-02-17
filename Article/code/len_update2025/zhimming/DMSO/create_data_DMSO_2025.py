#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 22:12:20 2025

@author: isabella
"""
import numpy as np
import pandas as pd
import re
import astropy.stats
import plotly.graph_objects as go
from astropy import units as u

pixelS = 0.065

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/zhimming/DMSO/figs/'


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

df_con = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/frames_DMSOcontrol.csv')
df_con['cell name'] = 'Control'
df_DSF = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/frames_DSF.csv')
df_DSF['cell name'] = 'DSF'
df_flg22 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/frames_flg22.csv')
df_flg22['cell name'] = 'flg22'


df_all = pd.concat([df_con,df_DSF,df_flg22], axis=0,ignore_index=True) 

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
cell_name_list = ['Control', 'DSF','flg22']
df_all['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,3)):
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
    print(len(df_all[(df_all['sort2']==1+4*m) & (df_all['cell name'] == i)]))
    print(len(df_all[(df_all['sort2']==2+4*m) & (df_all['cell name'] == i)]))
    print(len(df_all[(df_all['sort2']==3+4*m) & (df_all['cell name'] == i)]))
    print(len(df_all[(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)] ))


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

tm = ['cell1/','cell2/','cell3/','cell4/','cell5/']

pathsave_DMSO = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO/"
pathsave_DSF = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DSF/"
pathsave_flg22 = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/flg22/"

##########
# control
list_vals=[pathsave_DMSO+tm[0], pathsave_DMSO+tm[1], pathsave_DMSO+tm[2], pathsave_DMSO+tm[3]]
df_control= creation_of_data_ml(list_vals)
df_control['cell name'] = 'Control'
dens_control=creation_of_dens(list_vals)
dens_control['cell name'] = 'Control'
np.mean(dens_control['mean density'])
# DSF
list_vals=[pathsave_DSF+tm[1], pathsave_DSF+tm[2], pathsave_DSF+tm[3], pathsave_DSF+tm[4]]
df_10= creation_of_data_ml(list_vals)
df_10['cell name'] = 'DSF'
dens_10=creation_of_dens(list_vals)
dens_10['cell name'] = 'DSF'
np.mean(dens_10['mean density'])
# flg22
list_vals=[pathsave_flg22+tm[0], pathsave_flg22+tm[1], pathsave_flg22+tm[2], pathsave_flg22+tm[4]]
df_20= creation_of_data_ml(list_vals)
df_20['cell name'] = 'flg22'
dens_20=creation_of_dens(list_vals)
dens_20['cell name'] = 'flg22'
np.mean(dens_20['mean density'])


df_dens = pd.concat([dens_control,dens_10, dens_20], axis=0,ignore_index=True) 

df_allFullm = pd.concat([df_control,df_10, df_20], axis=0,ignore_index=True) 

cutval1 = 3.2164406150564564/pixelS
cutval2 = 6.148265832210858/pixelS
cutval3 = 16.262615814598842/pixelS
df_allFullm['cell name 2'] = df_allFullm['cell name'] 
df_allFullm['cell name 3'] = df_allFullm['cell name'] 
cell_name_list = ['Control', 'DSF','flg22']
df_allFullm['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,3)):
    print(i,m)
    df_allFullm['sort']=0
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
df_allFullm['mean filament movement']=df_allFullm['mean filament movement']*0.065/2
df_allFullm['mean intensity per length']=df_allFullm['mean intensity per length']/0.065

df_allFullm.to_csv(pathsave+'df_allFullm.csv')

###############################################################################
#
# create circular data
#
###############################################################################

def circ_stat(DataSet,pathsave,name):

    #data = np.asarray(pd_fil_info['filament angle'])*u.deg
    #weight = pd_fil_info['filament length']
    data = DataSet["mean filament angle"]*np.pi/180. # has to be in radians for astropy to work *u.deg
    weight = DataSet['mean filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))*180/np.pi # switch back to angles
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))

    return mean_angle,var_val


np.unique(df_allFullm['cell name 2'])

dmso_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control <3.22'],pathsave+'circ_angle/','Control_50')
dmso_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control 3.22-6.15'],pathsave+'circ_angle/','Control_100')
dmso_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control 6.15-16.26'],pathsave+'circ_angle/','Control_300')
dmso_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control >16.26'],pathsave+'circ_angle/','Control_M300')

dsf_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF <3.22'],pathsave+'circ_angle/','dsf_50')
dsf_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF 3.22-6.15'],pathsave+'circ_angle/','dsf_100')
dsf_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF 6.15-16.26'],pathsave+'circ_angle/','dsf_300')
dsf_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF >16.26'],pathsave+'circ_angle/','dsf_M300')

flg_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 <3.22'],pathsave+'circ_angle/','flg_50')
flg_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 3.22-6.15'],pathsave+'circ_angle/','flg_100')
flg_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 6.15-16.26'],pathsave+'circ_angle/','flg_300')
flg_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 >16.26'],pathsave+'circ_angle/','flg_M300')


namesD = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

list_dmso = [dmso_angle50[0],dmso_angle100[0],dmso_angle300[0],dmso_angleM300[0]]
list_dsf = [dsf_angle50[0],dsf_angle100[0],dsf_angle300[0],dsf_angleM300[0]]
list_flg = [flg_angle50[0],flg_angle100[0],flg_angle300[0],flg_angleM300[0]]

list_dmso1 = [dmso_angle50[1],dmso_angle100[1],dmso_angle300[1],dmso_angleM300[1]]
list_dsf1 = [dsf_angle50[1],dsf_angle100[1],dsf_angle300[1],dsf_angleM300[1]]
list_flg1 = [flg_angle50[1],flg_angle100[1],flg_angle300[1],flg_angleM300[1]] 



files = [list_dmso,list_dsf,list_flg]
filesV = np.asarray([list_dmso1,list_dsf1,list_flg1])
name_cell = ['Control','DSF','flg22']
angleFil=pd.DataFrame()
for i in range(len(name_cell)):
    dataang = {'angle' : files[i],'circular variance':filesV[i],'name': namesD,'cells': name_cell[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)

angleFil.to_csv(pathsave+'df_angles.csv')
