#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:40:30 2023

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import re
#import astropy.stats
import plotly.graph_objects as go
#from astropy import units as u
from statannotations.Annotator import Annotator
import scienceplots
import argparse
import matplotlib.ticker as ticker

import scienceplots

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

pathsave = '/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/figs/'

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

print('Upper: ',np.mean(df_all[df_all['cell name']=='Young']['mean angle change per filament']))
print('Middle: ',np.mean(df_all[df_all['cell name']=='Expanding']['mean angle change per filament']))
print('Bottom: ',np.mean(df_all[df_all['cell name']=='Mature']['mean angle change per filament']))

print('Upper: ',np.mean(df_all[df_all['cell name']=='Young']['mean length change per filament']))
print('Middle: ',np.mean(df_all[df_all['cell name']=='Expanding']['mean length change per filament']))
print('Bottom: ',np.mean(df_all[df_all['cell name']=='Mature']['mean length change per filament']))

###############################################################################
#
# histogram of length of filaments
#
###############################################################################

plt.close('all')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Young']*pixelS,bins=50, density=False,alpha=0.5,label='Upper')
#plt.xlim(0,400)
#plt.ylim(0,0.023)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_upper.png')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Expanding']*pixelS,bins=50, density=False,alpha=0.5,label='Middle')
#plt.xlim(0,1300)
#plt.ylim(0,0.023)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_middle.png')

plt.figure(figsize=(8.27/2, 2.5))
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Mature']*pixelS,bins=50, density=False,alpha=0.5,label='Bottom')
#plt.xlim(0,1300)
#plt.ylim(0,0.023)
plt.xlabel('Length [$\mu$m]')
plt.ylabel('Counts')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(pathsave+'hist_bottom.png')

fig, axd = plt.subplot_mosaic("ABC", figsize=(8.27,3))

axd['A'].hist(df_all["mean length per filament"][df_all['cell name']=='Young']*pixelS,bins=120, density=False,alpha=1,color='#FF6700',label='Upper')
axd['B'].hist(df_all["mean length per filament"][df_all['cell name']=='Expanding']*pixelS,bins=120, density=False,alpha=1,color='darkturquoise', label='Middle')
axd['C'].hist(df_all["mean length per filament"][df_all['cell name']=='Mature']*pixelS,bins=120, density=False,alpha=1, color='indigo',label='Bottom')

for nk in ['A','B','C']:
    threshold_list = [3.26, 6.51, 21.7]
    for s in threshold_list:
        axd[nk].axvline(x=s, color='red', linestyle='--', linewidth=.5, label=f'Threshold {s}')

axd['A'].set_xlabel('Length [$\mu$m]')
axd['B'].set_xlabel('Length [$\mu$m]')
axd['C'].set_xlabel('Length [$\mu$m]')

axd['A'].set_ylabel("Counts, Young")
axd['B'].set_ylabel("Counts, Expanding")
axd['C'].set_ylabel("Counts, Mature")
#axd['A'].sharey(axd['B'])
#axd['B'].sharey(axd['C'])

for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(pathsave + 'histograms_cotyledon.pdf')






rot_over = ['Young','Expanding','Mature']
for n in rot_over:
    data = np.array(df_all["mean length per filament"][df_all['cell name']==n]*pixelS)
    
    lengths = pd.DataFrame(data, columns=["length"])
    from sklearn.cluster import KMeans
    # Use K-means clustering to divide data into 4 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    lengths["group"] = kmeans.fit_predict(lengths)
    
    # Sort the clusters by their mean value
    cluster_means = lengths.groupby("group")["length"].mean().sort_values().index
    mapping = {cluster: i for i, cluster in enumerate(cluster_means)}
    lengths["group"] = lengths["group"].map(mapping)
    
    # Map groups to descriptive labels
    labels = {0: "Very Short", 1: "Short", 2: "Medium", 3: "Long"}
    lengths["group_label"] = lengths["group"].map(labels)
    
    # Plot the histogram with clusters
    plt.figure(figsize=(7, 4))
    for group, label in labels.items():
        plt.hist(lengths[lengths["group"] == group]["length"], bins=30, alpha=0.6, label=label)
    
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title("Length Distribution by Groups")
    plt.legend()
    plt.show()
    for k in range(3):
        print(n, np.max(lengths['length'][lengths['group']==k]), lengths['length'][lengths['group']==k].shape[0])
        
        
#################################
# creation into 4 groups for len change 
cutval1 = 15
cutval2 = 30
cutval3 = 100
df_all['cell name 2'] = df_all['cell name'] 
df_all['cell name 3'] = df_all['cell name'] 
cell_name_list = ['Young', 'Expanding','Mature']
df_all['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,4)):
    #print(i,m)
    df_all['sort']=0
    df_all['cell name 2'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i)) ] = '{0} <3.26'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i)) ] = '<3.26'
    df_all['sort2'].loc[((df_all['mean length per filament'] <= cutval1) & (df_all['cell name'] == i))] = 1 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = '{0} 3.26-6.51'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = '3.26-6.51'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval1) & (df_all['mean length per filament'] <= cutval2)  & (df_all['cell name'] == i))] = 2 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = '{0} 6.51-21.7'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = '6.51-21.7'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval2) & (df_all['mean length per filament'] <= cutval3)  & (df_all['cell name'] == i))] = 3 + 4*m
    
    df_all['cell name 2'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = '{0} >21.7'.format(i)
    df_all['cell name 3'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = '>21.7'
    df_all['sort2'].loc[((df_all['mean length per filament'] > cutval3) & (df_all['cell name'] == i))] = 4 + 4*m
    
    print(i)
    print(len(df_all[(df_all['sort2']==1+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==1+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==2+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==2+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==3+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==3+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))
    print(len(df_all[(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)]), len(df_all[(df_all['sort2']==4+4*m) & (df_all['cell name'] == i)])/len(df_all[df_all['cell name'] == i]))

df_all=df_all.sort_values(by=['sort2'])

# drop the smallest group in 20-30mins because contains 7 filaments
#df_all = df_all[df_all['sort2']!=16]

#####################
# change in angle and length per filament

x = "cell name 3"
hue = "cell name"
hue_order = ['Young', 'Expanding', 'Mature']
order = [ '<3.26', '3.26-6.51', '6.51-21.7', '>21.7']

pairs  = [
    [('<3.26', 'Mature'), ('<3.26', 'Expanding')],
     [('<3.26', 'Mature'), ('<3.26', 'Young')],
     [('<3.26', 'Expanding'), ('<3.26', 'Young')],
     [('3.26-6.51', 'Mature'), ('3.26-6.51', 'Expanding')],
     [('3.26-6.51', 'Mature'), ('3.26-6.51', 'Young')],
     [('3.26-6.51', 'Expanding'), ('3.26-6.51', 'Young')],
     [('6.51-21.7', 'Mature'), ('6.51-21.7', 'Expanding')],
     [('6.51-21.7', 'Mature'), ('6.51-21.7', 'Young')],
     [('6.51-21.7', 'Expanding'), ('6.51-21.7', 'Young')],
     [('>21.7', 'Mature'), ('>21.7', 'Expanding')],
     [('>21.7', 'Mature'), ('>21.7', 'Young')],
     [('>21.7', 'Expanding'), ('>21.7', 'Young')],
     ]

list_plots = ['mean angle change per filament','mean length change per filament','median length change per filament','mean length per filament']
list_ylabel = ['Mean angle change','Mean length change','Median length change','Mean filament length']

for i in range(len(list_plots)):  
    y = list_plots[i]
    plt.figure(figsize=(5,5))
    ax = sns.boxplot(data=df_all, x=x, y=y, order=order, hue=hue, hue_order=hue_order,showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "coral"},palette=cmap)
    annot = Annotator(ax, pairs, data=df_all, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(test='Mann-Whitney', verbose=2)
    annot.apply_test()
    annot.annotate()
    plt.ylabel(list_ylabel[i],size=10)
    plt.xlabel('')
    #plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups_stat.png'.format(list_ylabel[i]))


'''
#ylim=[[-16,16],[-70,65],[-32,26]]
for i in range(len(list_plots)):  
    print(i)
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df_all,y=df_all[list_plots[i]], x =df_all['cell name 3'],hue='cell name',showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "coral"},palette=cmap)
    plt.ylabel(list_ylabel[i],size=sizeL)
    plt.xlabel('')
    #plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups.png'.format(list_ylabel[i]))
'''


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
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i)) ] = '{0} <3.3'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i)) ] = '<3.3'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] <= cutval1) & (df_allFullm['cell name'] == i))] = 1 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = '{0} 3.3-6.5'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = '3.3-6.5'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval1) & (df_allFullm['mean filament length'] <= cutval2)  & (df_allFullm['cell name'] == i))] = 2 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = '{0} 6.5-21.7'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = '6.5-21.7'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval2) & (df_allFullm['mean filament length'] <= cutval3)  & (df_allFullm['cell name'] == i))] = 3 + 4*m
    
    df_allFullm['cell name 2'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = '{0} >21.7'.format(i)
    df_allFullm['cell name 3'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = '>21.7'
    df_allFullm['sort2'].loc[((df_allFullm['mean filament length'] > cutval3) & (df_allFullm['cell name'] == i))] = 4 + 4*m

df_allFullm=df_allFullm.sort_values(by=['sort2'])
df_allFullm['mean filament movement']=df_allFullm['mean filament movement']*0.217
df_allFullm['mean intensity per length']=df_allFullm['mean intensity per length']/0.217
df_allFullm['mean filament length']=df_allFullm['mean filament length']*0.217


df_allFullm.to_csv(pathsave+'df_allFullm.csv')

r_top_mid=np.load('/home/isabella/Documents/PLEN/dfs/data/5_time_dark_seedlings/timeseries/assortativity_vals.npy')

plt.figure()
plt.scatter(dens_top['mean density'],r_top_mid[:,0])
plt.scatter(dens_middle['mean density'],r_top_mid[:,1])
plt.scatter(dens_bottom['mean density'],r_top_mid[:,2])

###############################################################################
#
# plots in 4 groups
#
###############################################################################
plt.close('all')

x = "cell name 3"
hue = "cell name"
hue_order = ['Young', 'Expanding', 'Mature']
order = [ '<3.3', '3.3-6.5', '6.5-21.7', '>21.7']

pairs  = [
    [('<3.3', 'Mature'), ('<3.3', 'Expanding')],
     [('<3.3', 'Mature'), ('<3.3', 'Young')],
     [('<3.3', 'Expanding'), ('<3.3', 'Young')],
     [('3.3-6.5', 'Mature'), ('3.3-6.5', 'Expanding')],
     [('3.3-6.5', 'Mature'), ('3.3-6.5', 'Young')],
     [('3.3-6.5', 'Expanding'), ('3.3-6.5', 'Young')],
     [('6.5-21.7', 'Mature'), ('6.5-21.7', 'Expanding')],
     [('6.5-21.7', 'Mature'), ('6.5-21.7', 'Young')],
     [('6.5-21.7', 'Expanding'), ('6.5-21.7', 'Young')],
     [('>21.7', 'Mature'), ('>21.7', 'Expanding')],
     [('>21.7', 'Mature'), ('>21.7', 'Young')],
     [('>21.7', 'Expanding'), ('>21.7', 'Young')],
     ]


list_plots = ['mean filament bendiness','mean filament length','mean filament movement','filament movement per length','mean intensity','mean intensity per length','mean filament angle']
list_ylabel = ['Mean bendiness ratio','Mean length',r'Mean movement [$\mu m/s$]','Mean movement per length','Mean intensity',r'Mean intensity per $\mu m$','Mean filament angle']

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
    #plt.ylim(ylim[i])
    #plt.grid()
    #plt.xticks(rotation=45)
    
    if(i==0):
        plt.legend(fontsize=20,frameon=False)
    else:
        plt.legend().remove()
    ax.set_xlim(xmin=-0.5,xmax=3.5)
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups_stat.png'.format(list_plots[i]))



cell3 = np.unique(df_allFullm['cell name 2'])
for m in range(len(cell3)):
    print(cell3[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name 2']==cell3[m]]))
   
###############################################################################
#
# plots
#
###############################################################################
    
def barplot180(list_points,list_bins,save_dir,name_save,color_code):
    '''
    creation of circular barplot for angles

    Parameters
    ----------
    list_points : list
        list of binned weighted angles.
    list_bins : list
        list of bin values.
    save_dir : directory path
        path to save image.
    name_save : name
        name of figure.
    color_code : cmap
        value for cmap.

    Returns
    -------
    None.

    '''
    fig = go.Figure(go.Barpolar(
        r=list_points,
        theta=list_bins,
        width=5,
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.8,
        marker_color=color_code,
        # yellow '#FFFF00'
        # blue 	#0000FF
    ))
    fig.show()
    
    fig.update_layout(
        polar = dict(radialaxis = dict(showticklabels=False, ticks=''), sector = [0,180],
                     radialaxis_showgrid=False,
                     angularaxis=dict(
                         #showgrid=False,
                #rotation=180,
                #direction='clockwise',
                tickfont = dict(size = 30))
                             )
                )
    
    fig.write_image(save_dir+name_save, format='png')
    return


def circ_stat(DataSet,pathsave,name):

    #data = np.asarray(pd_fil_info['filament angle'])*u.deg
    #weight = pd_fil_info['filament length']
    data = DataSet["mean filament angle"]*u.deg
    weight = DataSet['mean filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))*180./np.pi
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))
        
    hist180,bins180 = np.histogram(0,int(180/5),[0,180])
        
    list_ec = np.zeros(len(bins180[1:]))
    for l in range(len(bins180[1:])):
    
        list_ec[l] = DataSet['mean filament length'][(DataSet["mean filament angle"]>bins180[l]) & (DataSet["mean filament angle"]<=bins180[l+1])].sum()
    
     
    bins180 = bins180[1:]-2.5
    name180 = name+"-stat.png"
    utilsF.barplot180(list_ec,bins180,pathsave,name180,color_code='#0000FF')
    return mean_angle,var_val

import sys
path="/home/isabella/Documents/PLEN/dfs/data/time_dark_seedlings/"
sys.path.append(path)

import utilsF

np.unique(df_allFullm['cell name 2'])

up_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young <3.3'],pathsave+'circ_angle/','Control_50')
up_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young 3.3-6.5'],pathsave+'circ_angle/','Control_100')
up_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young 6.5-21.7'],pathsave+'circ_angle/','Control_300')
up_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Young >21.7'],pathsave+'circ_angle/','Control_M300')

mid_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding <3.3'],pathsave+'circ_angle/','0-10_50')
mid_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding 3.3-6.5'],pathsave+'circ_angle/','0-10_100')
mid_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding 6.5-21.7'],pathsave+'circ_angle/','0-10_300')
mid_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Expanding >21.7'],pathsave+'circ_angle/','0-10_M300')

bot_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature <3.3'],pathsave+'circ_angle/','10-20_50')
bot_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature 3.3-6.5´'],pathsave+'circ_angle/','10-20_100')
bot_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature 6.5-21.7'],pathsave+'circ_angle/','10-20_300')
bot_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Mature >21.7'],pathsave+'circ_angle/','10-20_M300')


list_up = [up_angle50[0],up_angle100[0],up_angle300[0],up_angleM300[0]]
list_mid = [mid_angle50[0],mid_angle100[0],mid_angle300[0],mid_angleM300[0]]
list_bot = [bot_angle50[0],bot_angle100[0],bot_angle300[0],bot_angleM300[0]]

list_up1 = [up_angle50[1],up_angle100[1],up_angle300[1],up_angleM300[1]]
list_mid1 = [mid_angle50[1],mid_angle100[1],mid_angle300[1],mid_angleM300[1]]
list_bot1 = [bot_angle50[1],bot_angle100[1],bot_angle300[1],bot_angleM300[1]]

files = [list_up,list_mid,list_bot]
filesV = np.asarray([list_up1,list_mid1,list_bot1])
namesD = ['<3.3','3.3-6.5','6.5-21.7','>21.7']
name_cell = ['Young','Expanding','Mature']
angleFil=pd.DataFrame()
for i in range(len(name_cell)):
    dataang = {'angle' : files[i],'circular variance':filesV[i],'name': namesD,'cells': name_cell[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)
    
angleFil.to_csv(pathsave+'df_angles.csv')
###################
# plots
#
plt.figure(figsize=figsize)
sns.lineplot(data=angleFil, x='name', y='angle',hue='cells',marker='o',markersize=10,palette=sns.color_palette("Set2")[0:3])
plt.legend(fontsize=20,frameon=False)
plt.ylabel('Circular mean angle ['r'$\degree$]',fontsize=sizeL)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_all.png')


plt.figure(figsize=figsize)
sns.lineplot(data=angleFil, x='name', y='circular variance',hue='cells',marker='o',markersize=10,palette=sns.color_palette("Set2")[0:3])
plt.legend(fontsize=20,frameon=False)
plt.ylabel('Circular mean variance',fontsize=sizeL)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_var_all.png')

'''
###############################################################################
#
# PCA
#
###############################################################################
from sklearn.decomposition import PCA

df_allFullm.columns.tolist()

df_PCA = df_allFullm[['mean filament movement','mean filament length','mean filament angle', 'mean filament bendiness','mean intensity']]
df_PCA2 = df_allFullm[['filament movement per length','mean filament angle', 'mean filament bendiness','mean intensity per length']]

Y = df_allFullm['sort']
names = np.asarray(['Upper','Middle','Bottom'])
names2 = np.asarray(np.unique(df_allFullm['cell name 2']))
##########
# normalize data to zero mean
df_PCA_mean = np.mean(df_PCA2,axis=0)
X = df_PCA2 - df_PCA_mean

##########
# do PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print(pca.explained_variance_ratio_)
pca.explained_variance_
plt.figure(figsize=(12,8))
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], names):
    plt.scatter(
        X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1,fontsize=30)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'PCA.png')


plt.figure(figsize=(12,8))
for i, target_name in zip(np.arange(0,len(names2)), names2):
    plt.scatter(
        X_r[Y == i, 0], X_r[Y == i, 1],  alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1,fontsize=30)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'PCA.png')

np.cov(X_r.T)
pca.explained_variance_

print(abs( pca.components_ ))

from sklearn import datasets
iris = datasets.load_iris()

target_names = iris.target_names
y = iris.target
'''

###############################################################################
#
# final plot
#
###############################################################################


fig, axd = plt.subplot_mosaic("E..;HIJ", figsize=(8.27,4))


sns.lineplot(data=angleFil, x='name', y='angle',hue='cells',marker='o',markersize=10,palette=sns.color_palette("Set2")[0:3],ax=axd['E'])


axd['E'].legend(loc='best',frameon=False)



plt.legend(fontsize=12,frameon=False)
plt.ylabel('Circular mean angle ['r'$\degree$]',fontsize=sizeL)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()