#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:11:33 2023

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

pathsave = '/home/isabella/Documents/PLEN/dfs/data/Zhiming/31-07-23/code/output/DMSO_all/figs/'

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

print('control: ',np.mean(df_all[df_all['cell name']=='Control']['mean angle change per filament']))
print('DSF: ',np.mean(df_all[df_all['cell name']=='DSF']['mean angle change per filament']))
print('flg22: ',np.mean(df_all[df_all['cell name']=='flg22']['mean angle change per filament']))

print('control: ',np.mean(df_all[df_all['cell name']=='Control']['mean length change per filament']))
print('DSF: ',np.mean(df_all[df_all['cell name']=='DSF']['mean length change per filament']))
print('flg22: ',np.mean(df_all[df_all['cell name']=='flg22']['mean length change per filament']))

#################################
# creation into 4 groups for len change 
cutval1 = 50
cutval2 = 100
cutval3 = 300
df_all['cell name 2'] = df_all['cell name'] 
df_all['cell name 3'] = df_all['cell name'] 
cell_name_list = ['Control', 'DSF','flg22']
df_all['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,3)):
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

#####################
# change in angle and length per filament


list_plots = ['mean angle change per filament','mean length change per filament','median length change per filament']
list_ylabel = ['Mean angle change','Mean length change','Median length change']


for i in range(len(list_plots)):  
    print(i)
    plt.figure(figsize=figsize)
    sns.boxplot(data=df_all,y=df_all[list_plots[i]], x =df_all['cell name 3'],hue='cell name',showfliers = False,notch=True,bootstrap=10000,
                medianprops={"color": "crimson","linewidth":"2."},palette=cmap)
    plt.ylabel(list_ylabel[i],size=sizeL)
    plt.xlabel('')
    #plt.grid()
    #plt.xticks(rotation=45)
    plt.legend(fontsize=20).remove()
    plt.tight_layout()
    plt.savefig(pathsave+'groups/'+'{0} 4 groups.png'.format(list_ylabel[i]))

###############################################################################
#
# histogram of length of filaments
#
###############################################################################

plt.figure(figsize=figsize)
plt.hist(df_all["mean length per filament"][df_all['cell name']=='Control'],bins=50, density=True,alpha=0.5,label='Control')
#plt.yscale('log')
#plt.xlim(0,1250)
#plt.ylim(0,0.016)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(pathsave+'hist_control.png')

plt.figure(figsize=figsize)
plt.hist(df_all["mean length per filament"][df_all['cell name']=='DSF'],bins=50, density=True,alpha=0.5,label='DSF')
#plt.xlim(0,900)
#plt.ylim(0,0.016)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(pathsave+'hist_DSF.png')

plt.figure(figsize=figsize)
plt.hist(df_all["mean length per filament"][df_all['cell name']=='flg22'],bins=50, density=True,alpha=0.5,label='flg22')
#plt.xlim(0,900)
#plt.ylim(0,0.016)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(pathsave+'hist_flg22.png')

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

cutval1 = 50
cutval2 = 100
cutval3 = 300
df_allFullm['cell name 2'] = df_allFullm['cell name'] 
df_allFullm['cell name 3'] = df_allFullm['cell name'] 
cell_name_list = ['Control', 'DSF','flg22']
df_allFullm['sort2']=0
for i,m in zip(cell_name_list,np.arange(0,3)):
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
hue_order = ['Control', 'DSF','flg22']
order = [ '<3.3', '3.3-6.5', '6.5-19.5', '>19.5']

pairs  = [
    [('<3.3', 'Control'), ('<3.3', 'DSF')],
     [('<3.3', 'Control'), ('<3.3', 'flg22')],
     [('<3.3', 'DSF'), ('<3.3', 'flg22')],
     
     [('3.3-6.5', 'Control'), ('3.3-6.5', 'DSF')],
      [('3.3-6.5', 'Control'), ('3.3-6.5', 'flg22')],
      [('3.3-6.5', 'DSF'), ('3.3-6.5', 'flg22')],

     [('6.5-19.5', 'Control'), ('6.5-19.5', 'DSF')],
      [('6.5-19.5', 'Control'), ('6.5-19.5', 'flg22')],
      [('6.5-19.5', 'DSF'), ('6.5-19.5', 'flg22')],
     
     [('>19.5', 'Control'), ('>19.5', 'DSF')],
      [('>19.5', 'Control'), ('>19.5', 'flg22')],
      [('>19.5', 'DSF'), ('>19.5', 'flg22')],
     ]

plt.close('all')

list_plots = ['mean filament bendiness','mean filament length','mean filament movement','filament movement per length','mean intensity','mean intensity per length','mean filament angle']
list_ylabel = ['Mean bendiness ratio','Mean length','Mean movement [\u03bcm/s]','Mean movement per length','Mean intensity','Mean intensity per \u03bcm','Mean filament angle']

#ylim=[[-.1,100],[-0.05,1.4],[0,300]]

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
    
    if(i==2):
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
    

cell3 = np.unique(df_allFullm['cell name'])
for m in range(len(cell3)):
    print(cell3[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name']==cell3[m]]))
    
cell2 = np.unique(df_allFullm['cell name 2'])
for m in range(len(cell2)):
    print(cell2[m], np.mean(df_allFullm['mean filament length'][df_allFullm['cell name 2']==cell2[m]]))
   

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

dmso_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control <3.3'],pathsave+'circ_angle/','Control_50')
dmso_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control 3.3-6.5'],pathsave+'circ_angle/','Control_100')
dmso_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control 6.5-19.5'],pathsave+'circ_angle/','Control_300')
dmso_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='Control >19.5'],pathsave+'circ_angle/','Control_M300')

dsf_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF <3.3'],pathsave+'circ_angle/','dsf_50')
dsf_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF 3.3-6.5'],pathsave+'circ_angle/','dsf_100')
dsf_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF 6.5-19.5'],pathsave+'circ_angle/','dsf_300')
dsf_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='DSF >19.5'],pathsave+'circ_angle/','dsf_M300')

flg_angle50 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 <3.3'],pathsave+'circ_angle/','flg_50')
flg_angle100 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 3.3-6.5'],pathsave+'circ_angle/','flg_100')
flg_angle300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 6.5-19.5'],pathsave+'circ_angle/','flg_300')
flg_angleM300 = circ_stat(df_allFullm[df_allFullm['cell name 2']=='flg22 >19.5'],pathsave+'circ_angle/','flg_M300')


namesD = [ '<3.3', '3.3-6.5', '6.5-19.5', '>19.5']

list_dmso = [dmso_angle50[0],dmso_angle100[0],dmso_angle300[0],dmso_angleM300[0]]
list_dsf = [dsf_angle50[0],dsf_angle100[0],dsf_angle300[0],dsf_angleM300[0]]
list_flg = [flg_angle50[0],flg_angle100[0],flg_angle300[0],flg_angleM300[0]]

list_dmso1 = [dmso_angle50[1],dmso_angle100[1],dmso_angle300[1],dmso_angleM300[1]]
list_dsf1 = [dsf_angle50[1],dsf_angle100[1],dsf_angle300[1],dsf_angleM300[1]]
list_flg1 = [flg_angle50[1],flg_angle100[1],flg_angle300[1],flg_angleM300[1]] 



files = [list_dmso,list_dsf,list_flg]
filesV = np.asarray([list_dmso1,list_dsf1,list_flg1])
namesD = [ '<3.3', '3.3-6.5', '6.5-19.5', '>19.5']
name_cell = ['Control','DSF','flg22']
angleFil=pd.DataFrame()
for i in range(len(name_cell)):
    dataang = {'angle' : files[i],'circular variance':filesV[i],'name': namesD,'cells': name_cell[i]}
    frame = pd.DataFrame.from_dict(dataang)
    
    angleFil = pd.concat(([angleFil,frame]),ignore_index=True)

###################
# plots
#

plt.close('all')

size=20

plt.figure(figsize=figsize)
sns.scatterplot(data=angleFil, x='cells', y='angle',hue='cells',marker='o',alpha=0.7,s=200,palette=cmap[0:3])
sns.scatterplot(x='cells', y='angle',data=angleFil[angleFil['cells']=='Control'].groupby('cells', as_index=False)['angle'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='k')
sns.scatterplot(x='cells', y='angle',data=angleFil[angleFil['cells']=='DSF'].groupby('cells', as_index=False)['angle'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='k')
sns.scatterplot(x='cells', y='angle',data=angleFil[angleFil['cells']=='flg22'].groupby('cells', as_index=False)['angle'].mean(), s=250, alpha=1.,marker="+", label="Average light",color='k')
plt.legend(fontsize=20)
plt.ylabel('Circular mean angle ['r'$\degree$]',fontsize=sizeL)
plt.ylim(45,60)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_all.png')

plt.figure(figsize=figsize)
sns.scatterplot(data=angleFil, x='cells', y='circular variance',hue='cells',marker='o',alpha=0.7,s=200,palette=cmap[0:3])
sns.scatterplot(x='cells', y='circular variance',data=angleFil[angleFil['cells']=='Control'].groupby('cells', as_index=False)['circular variance'].mean(), s=250, alpha=1.,marker="+", label="Average control",color='k')
sns.scatterplot(x='cells', y='circular variance',data=angleFil[angleFil['cells']=='DSF'].groupby('cells', as_index=False)['circular variance'].mean(), s=250, alpha=1.,marker="+", label="Average DSF",color='k')
sns.scatterplot(x='cells', y='circular variance',data=angleFil[angleFil['cells']=='flg22'].groupby('cells', as_index=False)['circular variance'].mean(), s=250, alpha=1.,marker="+", label="Average flg22",color='k')
plt.legend(fontsize=25)
plt.ylabel('Circular variance',fontsize=sizeL)
#plt.ylim(0.01,0.015)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_var_all.png')


plt.figure(figsize=figsize)
sns.lineplot(data=angleFil, x='name', y='angle',hue='cells',marker='o',markersize=10,palette=cmap[0:3])

plt.legend(fontsize=20).remove()
plt.ylabel('Circular mean angle ['r'$\degree$]',fontsize=sizeL)
plt.xlabel('')
plt.legend(fontsize=20,loc='best',frameon=False)
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_allgroup.png')


plt.figure(figsize=figsize)
sns.lineplot(data=angleFil, x='name', y='circular variance',hue='cells',marker='o',markersize=10,palette=cmap[0:3])
plt.legend(fontsize=20).remove()
plt.ylabel('Circular variance',fontsize=sizeL)
plt.xlabel('')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pathsave+'groups/'+'circ_mean_var_allgrup.png')
