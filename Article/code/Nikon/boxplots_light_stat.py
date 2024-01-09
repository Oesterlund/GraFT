#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:21:07 202315

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import astropy.stats
import plotly.graph_objects as go
from astropy import units as u
from statannotations.Annotator import Annotator

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 

figsize = 9,6
sizeL=25
plt.close('all')

cmap=sns.color_palette("Set2")
sns.set_style("white")
sns.set_style("ticks")
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

###############################################################################
#
# import files
#
###############################################################################

pathsave = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/timeseries/figs/'

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
cutval1 = 50
cutval2 = 100
cutval3 = 300
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

df_all=df_all.sort_values(by=['sort2'])
df_all.to_csv('/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/light/'+'df_all.csv')

#####################
# change in angle and length per filament

x = "cell name 3"
hue = "cell name"
hue_order = ['Control', '0-10 mins', '10-20 mins','20-30 mins']
order = [ '<3.3', '3.3-6.5', '6.5-19.5', '>19.5']

pairs  = [
    [('<3.3', 'Control'), ('<3.3', '0-10 mins')],
     [('<3.3', 'Control'), ('<3.3', '10-20 mins')],
     [('<3.3', 'Control'), ('<3.3', '20-30 mins')],
     [('<3.3', '0-10 mins'), ('<3.3', '10-20 mins')],
     [('<3.3', '0-10 mins'), ('<3.3', '20-30 mins')],
     [('<3.3', '10-20 mins'), ('<3.3', '20-30 mins')],
     
     [('3.3-6.5', 'Control'), ('3.3-6.5', '0-10 mins')],
     [('3.3-6.5', 'Control'), ('3.3-6.5', '10-20 mins')],
     [('3.3-6.5', 'Control'), ('3.3-6.5', '20-30 mins')],
     [('3.3-6.5', '0-10 mins'), ('3.3-6.5', '10-20 mins')],
     [('3.3-6.5', '0-10 mins'), ('3.3-6.5', '20-30 mins')],
     [('3.3-6.5', '10-20 mins'), ('3.3-6.5', '20-30 mins')],

     [('6.5-19.5', 'Control'), ('6.5-19.5', '0-10 mins')],
     [('6.5-19.5', 'Control'), ('6.5-19.5', '10-20 mins')],
     [('6.5-19.5', 'Control'), ('6.5-19.5', '20-30 mins')],
     [('6.5-19.5', '0-10 mins'), ('6.5-19.5', '10-20 mins')],
     [('6.5-19.5', '0-10 mins'), ('6.5-19.5', '20-30 mins')],
     [('6.5-19.5', '10-20 mins'), ('6.5-19.5', '20-30 mins')],
     
     [('>19.5', 'Control'), ('>19.5', '0-10 mins')],
     [('>19.5', 'Control'), ('>19.5', '10-20 mins')],
     [('>19.5', 'Control'), ('>19.5', '20-30 mins')],
     [('>19.5', '0-10 mins'), ('>19.5', '10-20 mins')],
     [('>19.5', '0-10 mins'), ('>19.5', '20-30 mins')],
     [('>19.5', '10-20 mins'), ('>19.5', '20-30 mins')],
     ]

list_plots = ['mean angle change per filament','mean length change per filament','median length change per filament']
list_ylabel = ['Mean angle change','Mean length change','Median length change']

ylim=[[-16,16],[-70,65],[-32,26]]
for i in range(len(list_plots)):  
    y = list_plots[i]
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_all, x=x, y=y, order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "coral"},palette=cmap)
    annot = Annotator(ax, pairs, data=df_all, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test='Mann-Whitney', verbose=2,hide_non_significant=True)
    annot.apply_test()
    annot.annotate()
    plt.ylabel(list_ylabel[i],size=sizeL)
    plt.xlabel('')
    #plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    if(i==0):
        plt.legend(fontsize=20)
    else:
        plt.legend().remove()
    ax.set_xlim(xmin=-0.5,xmax=3.5)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups_stat.png'.format(list_ylabel[i]))

###############################################################################
#
# histogram of length of filaments
#
###############################################################################

plt.close('all')

plt.figure(figsize=figsize)
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Control'],bins=50, density=True,alpha=0.5,label='Control')
plt.xlim(0,1650)
plt.ylim(0,0.018)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(pathsave+'hist_control.png')

plt.figure(figsize=figsize)
plt.hist(df_all["mean length per filament"][df_all['cell name']=='0-10 mins'],bins=50, density=True,alpha=0.5,label='0-10 mins')
plt.xlim(0,1650)
plt.ylim(0,0.018)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(pathsave+'hist_0-10.png')

plt.figure(figsize=figsize)
plt.hist(df_all["mean length per filament"][df_all['cell name']=='10-20 mins'],bins=50, density=True,alpha=0.5,label='10-20 mins')
plt.xlim(0,1650)
plt.ylim(0,0.018)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(pathsave+'hist_10-20.png')

plt.figure(figsize=figsize)
plt.hist(df_all["mean length per filament"][df_all['cell name']=='20-30 mins'],bins=50, density=True,alpha=0.5,label='20-30 mins')
plt.xlim(0,1650)
plt.ylim(0,0.018)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(pathsave+'hist_20-30.png')

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

cutval1 = 50
cutval2 = 100
cutval3 = 300
df_allFullm['cell name 2'] = df_allFullm['cell name'] 
df_allFullm['cell name 3'] = df_allFullm['cell name'] 
cell_name_list = ['Control', '0-10 mins','10-20 mins','20-30 mins']
df_allFullm['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
    print(i,m)
    df_allFullm['sort']=0
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i)) ] = '{0} <3.3'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i)) ] = '<3.3'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i))] = 1 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = '{0} 3.3-6.5'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = '3.3-6.5'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = 2 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = '{0} 6.5-19.5'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = '6.5-19.5'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = 3 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = '{0} >19.5'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = '>19.5'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = 4 + 4*m
    
df_allFullm.to_csv(pathsave+'df_allFullm.csv')
df_allFullm=df_allFullm.sort_values(by=['sort2'])
df_allFullm['mean filament movement']=df_allFullm['mean filament movement']*0.065/2
df_allFullm['mean intensity per length']=df_allFullm['mean intensity per length']/0.065
###############################################################################
#
# plots in 4 groups
#
###############################################################################

x = "cell name 3"
hue = "cell name"
hue_order = ['Control', '0-10 mins', '10-20 mins','20-30 mins']
order = [ '<3.3', '3.3-6.5', '6.5-19.5', '>19.5']

pairs  = [
    [('<3.3', 'Control'), ('<3.3', '0-10 mins')],
     [('<3.3', 'Control'), ('<3.3', '10-20 mins')],
     [('<3.3', 'Control'), ('<3.3', '20-30 mins')],
     [('<3.3', '0-10 mins'), ('<3.3', '10-20 mins')],
     [('<3.3', '0-10 mins'), ('<3.3', '20-30 mins')],
     [('<3.3', '10-20 mins'), ('<3.3', '20-30 mins')],
     
     [('3.3-6.5', 'Control'), ('3.3-6.5', '0-10 mins')],
     [('3.3-6.5', 'Control'), ('3.3-6.5', '10-20 mins')],
     [('3.3-6.5', 'Control'), ('3.3-6.5', '20-30 mins')],
     [('3.3-6.5', '0-10 mins'), ('3.3-6.5', '10-20 mins')],
     [('3.3-6.5', '0-10 mins'), ('3.3-6.5', '20-30 mins')],
     [('3.3-6.5', '10-20 mins'), ('3.3-6.5', '20-30 mins')],

     [('6.5-19.5', 'Control'), ('6.5-19.5', '0-10 mins')],
     [('6.5-19.5', 'Control'), ('6.5-19.5', '10-20 mins')],
     [('6.5-19.5', 'Control'), ('6.5-19.5', '20-30 mins')],
     [('6.5-19.5', '0-10 mins'), ('6.5-19.5', '10-20 mins')],
     [('6.5-19.5', '0-10 mins'), ('6.5-19.5', '20-30 mins')],
     [('6.5-19.5', '10-20 mins'), ('6.5-19.5', '20-30 mins')],
     
     [('>19.5', 'Control'), ('>19.5', '0-10 mins')],
     [('>19.5', 'Control'), ('>19.5', '10-20 mins')],
     [('>19.5', 'Control'), ('>19.5', '20-30 mins')],
     [('>19.5', '0-10 mins'), ('>19.5', '10-20 mins')],
     [('>19.5', '0-10 mins'), ('>19.5', '20-30 mins')],
     [('>19.5', '10-20 mins'), ('>19.5', '20-30 mins')],
     ]


plt.close('all')

list_plots = ['mean filament movement','mean filament bendiness','mean intensity per length']
list_ylabel = ['Mean movement [\u03bcm/s]','Mean bendiness ratio','Mean intensity per \u03bcm']

ylim=[[-.1,120*0.07/2],[-0.05,1.43],[0,310/0.065]]

for i in range(len(list_plots)):  
    y = list_plots[i]
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_allFullm, x=x, y=y, order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "coral"},palette=cmap)
    annot = Annotator(ax, pairs, data=df_allFullm, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test='Mann-Whitney', verbose=2,fontsize=15,hide_non_significant=True)
    annot.apply_test()
    annot.annotate()
    plt.ylabel(list_ylabel[i],size=sizeL)
    plt.xlabel('')
    plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    
    if(i==0):
        plt.legend(fontsize=20,loc='best',frameon=False)
        #plt.ylim(-2,125)
    else:
        plt.legend().remove()
    ax.set_xlim(xmin=-0.5,xmax=3.5)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups_stat.png'.format(list_plots[i]))

for m in range(len(hue_order)):
    print(hue_order[m], np.mean(df_allFullm['mean filament movement'][df_allFullm['cell name']==hue_order[m]]))
    
cell2 = np.unique(df_allFullm['cell name 2'])
for m in range(len(cell2)):
    print(cell2[m], np.mean(df_allFullm['mean filament movement'][df_allFullm['cell name 2']==cell2[m]]))
    
for m in range(len(hue_order)):
    print(hue_order[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name']==hue_order[m]]))
    
for m in range(len(cell2)):
    print(cell2[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name 2']==cell2[m]]))
    