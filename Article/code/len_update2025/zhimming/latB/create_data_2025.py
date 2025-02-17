#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:53:33 2025

@author: isabella
"""
import numpy as np
import pandas as pd
import re
import astropy.stats
import plotly.graph_objects as go
from astropy import units as u
import networkx as nx
import pickle

pixelS = 0.065

pathsave = '/home/isabella/Documents/PLEN/dfs/data/len_update2025/zhimming/latB/figs/'

###############################################################################
#
# functions
#
###############################################################################

def creation_of_data_ml(list_vals,names):
    fullTrack=pd.DataFrame()
    for i in range(len(list_vals)):
        trackCur = pd.read_csv(list_vals[i]+'tracked_move.csv')
        tracklen = pd.read_csv(list_vals[i]+'tracked_filaments_info.csv')
        CurNam = names[i]
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
                'mean filament angle': mean_angle,'mean filament bendiness':filBend, 'mean intensity': filInt,'mean intensity per length':filtIntLen,
                'name': CurNam}
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

def circ_stat(DataSet):

    #data = np.asarray(pd_fil_info['filament angle'])*u.deg
    #weight = pd_fil_info['filament length']
    data = DataSet["mean filament angle"]*np.pi/180. # has to be in radians for astropy to work *u.deg
    weight = DataSet['mean filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))*180/np.pi # switch back to angles
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))

    return mean_angle,var_val

###############################################################################
#
# Latb Light treatment
#
###############################################################################


###############################################################################
#
# import files
#
###############################################################################

df_DC = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_control.csv')
df_DC['cell name'] = 'Control'
df_D10 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_10.csv')
df_D10['cell name'] = '0-10 mins'
df_D20 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_20.csv')
df_D20['cell name'] = '10-20 mins'
df_D30 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/frames_30.csv')
df_D30['cell name'] = '20-30 mins'

df_all = pd.concat([df_DC,df_D10,df_D20,df_D30], axis=0,ignore_index=True) 

##### change in angle per filament
l_mAL = []
std_l_mAL = np.zeros(len(df_all))
mean_l_mAL = np.zeros(len(df_all))
median_l_mAL = np.zeros(len(df_all))
mean_l_mlL = np.zeros(len(df_all))
l_mlL = []
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
    l_mlL.append(mlL)
    median_l_mlL[l] = np.median(mlL)
    
df_all['mean angle change per filament']=mean_l_mAL
df_all['median angle change per filament']=median_l_mAL
df_all['std of mean angle change per filament'] = std_l_mAL

df_all['mean length change per filament']=mean_l_mlL
df_all['mean length change per filament full']=l_mlL
df_all['median length change per filament']=median_l_mlL

print('control: ',np.mean(df_all[df_all['cell name']=='Control']['mean angle change per filament']))
print('0-10: ',np.mean(df_all[df_all['cell name']=='0-10 mins']['mean angle change per filament']))
print('10-20: ',np.mean(df_all[df_all['cell name']=='10-20 mins']['mean angle change per filament']))
print('20-30: ',np.mean(df_all[df_all['cell name']=='20-30 mins']['mean angle change per filament']))

print('control: ',np.mean(df_all[df_all['cell name']=='Control']['mean length change per filament']))
print('0-10: ',np.mean(df_all[df_all['cell name']=='0-10 mins']['mean length change per filament']))
print('10-20: ',np.mean(df_all[df_all['cell name']=='10-20 mins']['mean length change per filament']))
print('20-30: ',np.mean(df_all[df_all['cell name']=='20-30 mins']['mean length change per filament']))

#################################
# creation into 4 groups for len change 
cutval1 = 3.2164406150564564/pixelS
cutval2 = 6.148265832210858/pixelS
cutval3 = 16.262615814598842/pixelS
df_all['cell name 2'] = df_all['cell name'] 
df_all['cell name 3'] = df_all['cell name'] 
cell_name_list = ['Control', '0-10 mins','10-20 mins','20-30 mins']
df_all['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
    print(i,m)
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
df_all.to_csv(pathsave + 'df_all_light.csv')



###############################################################################
#
# division into 4 groups for creation of data ml
#
###############################################################################

plant=['control/','latB_0-10min/','latB_10-20min/','latB_20-30min/']
tm = ['cell1/','cell2/','cell3/','cell4/']
names =['cell 1','cell 2','cell 3', 'cell 4']
pathsave2 = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/'

list_vals=[pathsave2+plant[0]+tm[0], pathsave2+plant[0]+tm[1], pathsave2+plant[0]+tm[2], pathsave2+plant[0]+tm[3]]
df_control= creation_of_data_ml(list_vals,names)
df_control['cell name'] = 'Control'
dens_control=creation_of_dens(list_vals)
dens_control['cell name'] = 'Control'
np.mean(dens_control['mean density'])
# middle
list_vals=[pathsave2+plant[1]+tm[0], pathsave2+plant[1]+tm[1], pathsave2+plant[1]+tm[2], pathsave2+plant[1]+tm[3]]
df_10= creation_of_data_ml(list_vals,names)
df_10['cell name'] = '0-10 mins'
dens_10=creation_of_dens(list_vals)
dens_10['cell name'] = '0-10 mins'
np.mean(dens_10['mean density'])
# bottom
list_vals=[pathsave2+plant[2]+tm[0], pathsave2+plant[2]+tm[1], pathsave2+plant[2]+tm[2], pathsave2+plant[2]+tm[3]]
df_20= creation_of_data_ml(list_vals,names)
df_20['cell name'] = '10-20 mins'
dens_20=creation_of_dens(list_vals)
dens_20['cell name'] = '10-20 mins'
np.mean(dens_20['mean density'])
# 20-30 mins
list_vals=[pathsave2+plant[3]+tm[0], pathsave2+plant[3]+tm[1], pathsave2+plant[3]+tm[2], pathsave2+plant[3]+tm[3]]
df_30= creation_of_data_ml(list_vals,names)
df_30['cell name'] = '20-30 mins'
dens_30=creation_of_dens(list_vals)
dens_30['cell name'] = '20-30 mins'
np.mean(dens_30['mean density'])

df_dens = pd.concat([dens_control,dens_10, dens_20,dens_30], axis=0,ignore_index=True) 

df_allFullm = pd.concat([df_control,df_10, df_20,df_30], axis=0,ignore_index=True) 

cutval1 = 3.2164406150564564/pixelS
cutval2 = 6.148265832210858/pixelS
cutval3 = 16.262615814598842/pixelS
df_allFullm['cell name 2'] = df_allFullm['cell name'] 
df_allFullm['cell name 3'] = df_allFullm['cell name'] 
cell_name_list = ['Control', '0-10 mins','10-20 mins','20-30 mins']
df_allFullm['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
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

df_allFullm.to_csv(pathsave+'df_allFullm_light.csv')


###############################################################################
#
# circular stat
#
###############################################################################

np.unique(df_allFullm['cell name 2'])

typeL = ['Control', '0-10 mins','10-20 mins','20-30 mins'] 
list_A = []
list_V = []
for n in range(len(typeL)):
    
    angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} <3.22'.format(typeL[n])])
    angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} 3.22-6.15'.format(typeL[n])])
    angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} 6.15-16.26'.format(typeL[n])])
    angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} >16.26'.format(typeL[n])])


    namesD = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

    list_A.extend([angle50[0],angle100[0],angle300[0],angleM300[0]])
    list_V.extend([angle50[1].item(),angle100[1].item(),angle300[1].item(),angleM300[1].item()])
    

namesD = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']
angleFil=pd.DataFrame()
for i in range(len(typeL)):
    dataang = {'angle' : list_A[0+4*i:4+4*i],'circular variance':list_V[0+4*i:4+4*i],'name': namesD,'cells': typeL[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)

angleFil.to_csv(pathsave+'df_angles_light.csv')


###############################################################################
#
# Latb Dark treatment
#
###############################################################################


def creation_of_data_ml(list_vals):
    fullTrack=pd.DataFrame()
    for i in range(len(list_vals)):
        trackCur = pd.read_csv(list_vals[i]+'tracked_move.csv')
        tracklen = pd.read_csv(list_vals[i]+'tracked_filaments_info.csv')
        tagsU = tracklen['filament'].unique()
        move_pL = []
        move_pLF = []
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
                move_pLF.append(move/fil[0:len(move)])
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
                'mean filament angle': mean_angle,'mean filament bendiness':filBend, 'mean intensity': filInt,'mean intensity per length':filtIntLen,
                'filament movement per length all': move_pLF}
        frame = pd.DataFrame.from_dict(data)
        
        fullTrack = pd.concat(([fullTrack,frame]),ignore_index=True)
    return fullTrack

###############################################################################
#
# import files
#
###############################################################################

df_DC = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_control.csv')
df_DC['cell name'] = 'Control'
df_D10 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_10.csv')
df_D10['cell name'] = '0-10 mins'
df_D20 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_20.csv')
df_D20['cell name'] = '10-20 mins'
df_D30 = pd.read_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/frames_30.csv')
df_D30['cell name'] = '20-30 mins'

df_all = pd.concat([df_DC,df_D10,df_D20,df_D30], axis=0,ignore_index=True) 

##### change in angle per filament
l_mAL = []
std_l_mAL = np.zeros(len(df_all))
mean_l_mAL = np.zeros(len(df_all))
median_l_mAL = np.zeros(len(df_all))
mean_l_mlL = np.zeros(len(df_all))
l_mlL = []
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
    l_mlL.append(mlL)
    median_l_mlL[l] = np.median(mlL)
    
    

df_all['mean angle change per filament']=mean_l_mAL
df_all['median angle change per filament']=median_l_mAL
df_all['std of mean angle change per filament'] = std_l_mAL

df_all['mean length change per filament']=mean_l_mlL
df_all['mean length change per filament full']=l_mlL
df_all['median length change per filament']=median_l_mlL

print('control: ',np.mean(df_all[df_all['cell name']=='Control']['mean angle change per filament']))
print('0-10: ',np.mean(df_all[df_all['cell name']=='0-10 mins']['mean angle change per filament']))
print('10-20: ',np.mean(df_all[df_all['cell name']=='10-20 mins']['mean angle change per filament']))
print('20-30: ',np.mean(df_all[df_all['cell name']=='20-30 mins']['mean angle change per filament']))

print('control: ',np.mean(df_all[df_all['cell name']=='Control']['mean length change per filament']))
print('0-10: ',np.mean(df_all[df_all['cell name']=='0-10 mins']['mean length change per filament']))
print('10-20: ',np.mean(df_all[df_all['cell name']=='10-20 mins']['mean length change per filament']))
print('20-30: ',np.mean(df_all[df_all['cell name']=='20-30 mins']['mean length change per filament']))

#################################
# creation into 4 groups for len change 
cutval1 = 3.2164406150564564/pixelS
cutval2 = 6.148265832210858/pixelS
cutval3 = 16.262615814598842/pixelS
df_all['cell name 2'] = df_all['cell name'] 
df_all['cell name 3'] = df_all['cell name'] 
cell_name_list = ['Control', '0-10 mins','10-20 mins','20-30 mins']
df_all['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
    print(i,m)
    df_all['sort']=0
    df_all['cell name 2'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i)) ] = '{0} <3.3'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i)) ] = '<3.3'
    df_all['sort2'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i))] = 1 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = '{0} 3.3-6.5'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = '3.3-6.5'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = 2 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = '{0} 6.5-19.5'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = '6.5-19.5'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = 3 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = '{0} >19.5'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = '>19.5'
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

# drop the smallest group in 20-30mins because contains 7 filaments
df_all = df_all[df_all['sort2']!=16]
df_all.to_csv(pathsave + 'df_all_dark.csv')

###############################################################################
#
# division into 4 groups for creation of data ml
#
###############################################################################

plant=['control/','latB_0-10min/','latB_10-20min/','latB_20-30min/']
tm = ['cell1/','cell2/','cell3/','cell4/']

pathsave2 = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/'

list_vals=[pathsave2+plant[0]+tm[0], pathsave2+plant[0]+tm[1], pathsave2+plant[0]+tm[2], pathsave2+plant[0]+tm[3]]
df_control= creation_of_data_ml(list_vals)
df_control['cell name'] = 'Control'
dens_control=creation_of_dens(list_vals)
dens_control['cell name'] = 'Control'
np.mean(dens_control['mean density'])
# middle
list_vals=[pathsave2+plant[1]+tm[0], pathsave2+plant[1]+tm[1], pathsave2+plant[1]+tm[2], pathsave2+plant[1]+tm[3]]
df_10= creation_of_data_ml(list_vals)
df_10['cell name'] = '0-10 mins'
dens_10=creation_of_dens(list_vals)
dens_10['cell name'] = '0-10 mins'
np.mean(dens_10['mean density'])
# bottom
list_vals=[pathsave2+plant[2]+tm[0], pathsave2+plant[2]+tm[1], pathsave2+plant[2]+tm[2], pathsave2+plant[2]+tm[3]]
df_20= creation_of_data_ml(list_vals)
df_20['cell name'] = '10-20 mins'
dens_20=creation_of_dens(list_vals)
dens_20['cell name'] = '10-20 mins'
np.mean(dens_20['mean density'])
# 20-30 mins
list_vals=[pathsave2+plant[3]+tm[0], pathsave2+plant[3]+tm[1], pathsave2+plant[3]+tm[2], pathsave2+plant[3]+tm[3]]
df_30= creation_of_data_ml(list_vals)
df_30['cell name'] = '20-30 mins'
dens_30=creation_of_dens(list_vals)
dens_30['cell name'] = '20-30 mins'
np.mean(dens_30['mean density'])

df_dens = pd.concat([dens_control,dens_10, dens_20,dens_30], axis=0,ignore_index=True) 

df_allFullm = pd.concat([df_control,df_10, df_20,df_30], axis=0,ignore_index=True) 


cutval1 = 3.2164406150564564/pixelS
cutval2 = 6.148265832210858/pixelS
cutval3 = 16.262615814598842/pixelS
df_allFullm['cell name 2'] = df_allFullm['cell name'] 
df_allFullm['cell name 3'] = df_allFullm['cell name'] 
cell_name_list = ['Control', '0-10 mins','10-20 mins','20-30 mins']
df_allFullm['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
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

df_allFullm2 = df_allFullm.copy()
df_allFullm['mean filament movement']=df_allFullm['mean filament movement']*0.065/2
df_allFullm['mean intensity per length']=df_allFullm['mean intensity per length']/0.065
# drop the smallest group in 20-30mins because contains 13 filaments
df_allFullm = df_allFullm[df_allFullm['sort2']!=16]
df_allFullm.to_csv(pathsave+'df_allFullm_dark.csv')


###############################################################################
#
# create circular data
#
###############################################################################
print('circular val for dark latB')
np.unique(df_allFullm['cell name 2'])

typeL = ['Control', '0-10 mins','10-20 mins','20-30 mins'] 
list_A = []
list_V = []
for n in range(len(typeL)):
    
    angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} <3.22'.format(typeL[n])])
    angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} 3.22-6.15'.format(typeL[n])])
    angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} 6.15-16.26'.format(typeL[n])])
    angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='{0} >16.26'.format(typeL[n])])


    namesD = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']

    list_A.extend([angle50[0],angle100[0],angle300[0],angleM300[0]])
    list_V.extend([angle50[1].item(),angle100[1].item(),angle300[1].item(),angleM300[1].item()])
    

namesD = [ '<3.22', '3.22-6.15', '6.15-16.26', '>16.26']
angleFil=pd.DataFrame()
for i in range(len(typeL)):
    dataang = {'angle' : list_A[0+4*i:4+4*i],'circular variance':list_V[0+4*i:4+4*i],'name': namesD,'cells': typeL[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)

angleFil.to_csv(pathsave+'df_angles_dark.csv')

###############################################################################
#
# assortativity
#
###############################################################################

def circ_stat2(DataSet):

    #data = np.asarray(pd_fil_info['filament angle'])*u.deg
    #weight = pd_fil_info['filament length']
    data = DataSet["filament angle"]*np.pi/180. # has to be in radians for astropy to work *u.deg
    weight = DataSet['filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))*180/np.pi # switch back to angles
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))
        
    return mean_angle,var_val

assFil=pd.DataFrame()
for i in range(2):
    if(i==0):
        pathsavel = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/"
        pathsaveFig = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/figs/'
        name = 'Light'
        
    else:
        pathsavel = "/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/"
        pathsaveFig = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/dark/timeseries/figs/'
        name = 'Dark'
    plant=['control/','latB_0-10min/','latB_10-20min/','latB_20-30min/']
    tm = ['cell1/','cell2/','cell3/','cell4/']
    
    
    r_all=[]
    density = []
    angle = []
    angleVar = []
    
    for c in range(len(plant)):
        for m in range(len(tm)):
            file = open(pathsavel+plant[c]+tm[m]+'tagged_graph.gpickle','rb')
            tagged_graph = pickle.load(file)
            pd_fil_info = pd.read_csv(pathsavel+plant[c]+tm[m]+'tracked_filaments_info.csv')
            r_db=[]
            dens_db = []
            for i in range(len(tagged_graph)):
                r_db=np.append(r_db,nx.degree_assortativity_coefficient(tagged_graph[i]))
            
            dens_db = np.unique(pd_fil_info['filament density'])
            
            angleD,varD = circ_stat2(pd_fil_info)
            
            density=np.append(density,np.mean(dens_db))
            r_all=np.append(r_all,np.mean(r_db))
            angle=np.append(angle,angleD)
            angleVar=np.append(angleVar,varD)
    
    r_allC = r_all.reshape((4, 4))
    densityC = density.reshape((4, 4))
    angleC = angle.reshape((4,4))
    angleVarC = angleVar.reshape((4,4))
    
    
    #name_cell = ['Control','0-10 mins','10-20 mins','20-30 mins']
    name_cell = ['Control','0-10','10-20','20-30']
    for i in range(len(name_cell)):
        name_cellN = name_cell[i]+' {0}'.format(name)
        dataang = {'Assortativity' : r_allC[i],'Cell density':densityC[i], 'Mean circular angle': angleC[i],'Mean angle var':angleVarC[i],'cells': name_cellN,'Treatment':name_cell[i],'Type':name}
        frame = pd.DataFrame.from_dict(dataang)
        
        assFil = pd.concat(([assFil,frame]),ignore_index=True)
        

assFil.to_csv(pathsave+'df_assFil.csv')
