#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: isabella
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import astropy.stats
import scienceplots
from scipy import stats
import matplotlib as mpl
plt.style.use(['science','nature']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('axes', labelsize=10)
sizeL=10
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}

plt.rcParams.update(params)


mpl.rcParams['text.usetex'] = False  # Disable LaTeX
mpl.rcParams['axes.unicode_minus'] = True  # Ensure minus sign is rendered correctly


cmap=sns.color_palette("colorblind")

overpath = 'define_this_path'

###############################################################################
#
# functions
#
###############################################################################

savepathAll = overpath+'/noise001/pooled/'

###############################################################################
#
# load files in
#
###############################################################################

#############
# 40 lines data
overpath40 = overpath+'/noise001/noise40/'
savepathLines40 = overpath40 + 'line_comparisons/'

linesAn40 = ['0_df_line_comparison','1_df_line_comparison','2_df_line_comparison','3_df_line_comparison','4_df_line_comparison']

dfLinesAn40 = pd.DataFrame()
df_JI40 = pd.DataFrame()
index=0
for kl in range(len(linesAn40)):
    
    dfLinesAnIn = pd.read_csv(savepathLines40+'{0}_df_line_comparison.csv'.format(kl))
    dfLinesAn40 = pd.concat([dfLinesAn40, dfLinesAnIn], ignore_index=True)

    df_JIIn = pd.read_csv(overpath40+'JI/'+'{0}_JI.csv'.format(kl))
    
    if(index>0):
        df_JIIn['frame']=df_JIIn['frame']+index+1
    
    index=np.max(df_JIIn['frame'])
    
    df_JI40 = pd.concat([df_JI40, df_JIIn], ignore_index=True)
    
#dfLinesAn40 = pd.read_csv(savepathLines40+'4_df_line_comparison.csv')  
df_densImage40 = pd.read_csv(overpath+'noise001/lines/density/' + 'density40.csv') 

#############
# 30 lines data
overpath30 = overpath+'/noise001/noise30/'
savepathLines30 = overpath30 + 'line_comparisons/'

linesAn30 = ['0_df_line_comparison','1_df_line_comparison','2_df_line_comparison','3_df_line_comparison','4_df_line_comparison']

dfLinesAn30 = pd.DataFrame()
df_JI30 = pd.DataFrame()
index=0
for kl in range(len(linesAn30)):
    
    dfLinesAnIn = pd.read_csv(savepathLines30+'{0}_df_line_comparison.csv'.format(kl))
    dfLinesAn30 = pd.concat([dfLinesAn30, dfLinesAnIn], ignore_index=True)
    
    df_JIIn = pd.read_csv(overpath30 + 'JI/'+'{0}_JI.csv'.format(kl))
    
    if(index>0):
        df_JIIn['frame']=df_JIIn['frame']+index+1
    
    index=np.max(df_JIIn['frame'])
    
    df_JI30 = pd.concat([df_JI30, df_JIIn], ignore_index=True)
    
#dfLinesAn30 = pd.read_csv(savepathLines30+'4_df_line_comparison.csv')  
df_densImage30 = pd.read_csv(overpath+'/noise001/lines/density/' + 'density30.csv') 

#############
# 20 lines data
overpath20 = overpath+'/noise001/noise20/'
savepathLines20 = overpath20 + 'line_comparisons/'

linesAn = ['0_df_line_comparison','1_df_line_comparison','2_df_line_comparison','3_df_line_comparison','4_df_line_comparison']

dfLinesAn20 = pd.DataFrame()
df_JI20 = pd.DataFrame()
index=0
for kl in range(len(linesAn)):
    
    dfLinesAnIn = pd.read_csv(savepathLines20+'{0}_df_line_comparison.csv'.format(kl))
    dfLinesAn20 = pd.concat([dfLinesAn20, dfLinesAnIn], ignore_index=True)
    
    df_JIIn = pd.read_csv(overpath20 + 'JI/'+'{0}_JI.csv'.format(kl))
    
    if(index>0):
        df_JIIn['frame']=df_JIIn['frame']+index+1
    
    index=np.max(df_JIIn['frame'])
    
    df_JI20 = pd.concat([df_JI20, df_JIIn], ignore_index=True)
  
#dfLinesAn20 = pd.read_csv(savepathLines20+'4_df_line_comparison.csv')
df_densImage20 = pd.read_csv (overpath+'/noise001/lines/density/' + 'density20.csv') 

#############
# 10 lines data
overpath10 = overpath+'/noise001/noise10/'
savepathLines10 = overpath10+'line_comparisons/'

linesAn10 = ['0_df_line_comparison','1_df_line_comparison','2_df_line_comparison','3_df_line_comparison','4_df_line_comparison']#,
           #'5_df_line_comparison','6_df_line_comparison','7_df_line_comparison','8_df_line_comparison','9_df_line_comparison']

dfLinesAn10 = pd.DataFrame()
df_JI10 = pd.DataFrame()
index=0
for kl in range(len(linesAn10)):
    
    dfLinesAnIn10 = pd.read_csv(savepathLines10+'{0}_df_line_comparison.csv'.format(kl))
    dfLinesAn10 = pd.concat([dfLinesAn10, dfLinesAnIn10], ignore_index=True)
    
    df_JIIn10 = pd.read_csv(overpath10+'JI/'+'{0}_JI.csv'.format(kl))
    if(index>0):
        df_JIIn10['frame']=df_JIIn10['frame']+index+1
    
    index=np.max(df_JIIn10['frame'])
    df_JI10 = pd.concat([df_JI10, df_JIIn10], ignore_index=True)

#dfLinesAn10 = pd.read_csv(savepathLines10+'4_df_line_comparison.csv')
df_densImage10 = pd.read_csv ('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/noise001/lines/density/' + 'density10.csv') 

###############################################################################
# 5 lines data
overpath5 = overpath+'/noise001/noise5/'

savepathLines5 = overpath5+'line_comparisons/'

linesAn5 = ['0_df_line_comparison','1_df_line_comparison','2_df_line_comparison','3_df_line_comparison','4_df_line_comparison']

dfLinesAn5 = pd.DataFrame()
df_JI5 = pd.DataFrame()
index=0
for kl in range(len(linesAn5)):
    
    dfLinesAnIn5 = pd.read_csv(savepathLines5+'{0}_df_line_comparison.csv'.format(kl))
    dfLinesAn5 = pd.concat([dfLinesAn5, dfLinesAnIn5], ignore_index=True)
    
    df_JIIn5 = pd.read_csv(overpath5+'JI/'+'{0}_JI.csv'.format(kl))
    if(index>0):
        df_JIIn5['frame']=df_JIIn5['frame']+index+1
    
    index=np.max(df_JIIn5['frame'])
    df_JI5 = pd.concat([df_JI5, df_JIIn5], ignore_index=True)
    
#dfLinesAn5 = pd.read_csv(savepathLines5+'4_df_line_comparison.csv')

df_densImage5 = pd.read_csv (overpath+'noise001/lines/density/' + 'density5.csv') 


df_densImage5['line type'] = 5
df_densImage10['line type'] = 10
df_densImage20['line type'] = 20
df_densImage30['line type'] = 30
df_densImage40['line type'] = 40

###############################################################################
#
# Group up based on image density
#
###############################################################################

bins = np.linspace(0, 0.022, 100)  # 6 edges, 5 bins
plt.figure(figsize=(8.27,5))
# Plot histograms
plt.hist(df_densImage40['Density full image'], bins=bins, alpha=0.8, label='40 lines data')
plt.hist(df_densImage30['Density full image'], bins=bins, alpha=0.8, label='30 lines data')
plt.hist(df_densImage20['Density full image'], bins=bins, alpha=0.8, label='20 lines data')
plt.hist(df_densImage10['Density full image'], bins=bins, alpha=0.8, label='10 lines data')
plt.hist(df_densImage5['Density full image'], bins=bins, alpha=0.8, label='5 lines data')
plt.legend()
plt.tight_layout()
#plt.savefig(savepathAll+'density_image40+30+20+10+5.png')

#binsImDens = [0, 0.002, 0.003, 0.004, 0.005,0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013,0.2]
binsImDens = [0.00, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014,0.016,0.018,0.02, 1]

labelsIm = np.arange(len(binsImDens)-1)

# Use pd.cut to create new column
df_densImage40['density groups image'] = pd.cut(df_densImage40['Density full image'], bins=binsImDens, labels=labelsIm, right=False)
df_densImage40['density groups image'].value_counts().sort_index()


df_densImage30['density groups image'] = pd.cut(df_densImage30['Density full image'], bins=binsImDens, labels=labelsIm, right=False)
df_densImage30['density groups image'].value_counts().sort_index()


df_densImage20['density groups image'] = pd.cut(df_densImage20['Density full image'], bins=binsImDens, labels=labelsIm, right=False)
df_densImage20['density groups image'].value_counts().sort_index()


df_densImage10['density groups image'] = pd.cut(df_densImage10['Density full image'], bins=binsImDens, labels=labelsIm, right=False)
df_densImage10['density groups image'].value_counts().sort_index()


df_densImage5['density groups image'] = pd.cut(df_densImage5['Density full image'], bins=binsImDens, labels=labelsIm, right=False)
df_densImage5['density groups image'].value_counts().sort_index()
###############################################################################
# merge all pandas together based on line data

#####################
# 40 lines
df_ALLIn40 = dfLinesAn40.merge(df_JI40,on='frame', how='inner')
df_ALL40 = df_ALLIn40.merge(df_densImage40,on='frame', how='inner')

FSList40 = np.zeros(501)
overlapList40 = np.zeros(501)
cordefList40 = np.zeros(501)
nooverlapList40 = np.zeros(501)
groupL40 = np.zeros(501)
dens40 = np.zeros(501)
JI40 = np.zeros(501)
f40 = np.zeros(501)
t40 = np.zeros(501).astype(object)
for i in np.unique(dfLinesAn40['frame']):
    dfCur40 = df_ALL40[df_ALL40['frame']==i]
    # remove all places where we look at the true that has not been matched with defined
    df_ALL40 = df_ALL40[pd.notna(df_ALL40['match index'])]
    FSList40[i] = np.median(df_ALL40['FS_coverage'][df_ALL40['frame']==i])
    
    overlapList40[i] = np.median(df_ALL40['overlap ratio'][df_ALL40['frame']==i])
    cordefList40[i] = len(np.unique(dfCur40['true index'][pd.notna(dfCur40['true index'])])) / len(np.unique(dfCur40['match index'][pd.notna(dfCur40['match index'])]))

    nooverlapList40[i] = len(dfCur40['overlap ratio'][(dfCur40['overlap ratio']==0)] ) / len(np.unique(dfCur40['true index'][pd.notna(dfCur40['true index'])])) 
    
    groupL40[i] = np.unique(df_ALL40['density groups image'][df_ALL40['frame']==i])[0]
    dens40[i] = np.unique(df_ALL40['Density full image'][df_ALL40['frame']==i])[0]
    JI40[i] = np.unique(df_ALL40['JI'][df_ALL40['frame']==i])[0]
    f40[i] = i
    t40[i] = 'line 40'
#####################
# 30 lines
df_ALLIn30 = dfLinesAn30.merge(df_JI30,on='frame', how='inner')
df_ALL30 = df_ALLIn30.merge(df_densImage30,on='frame', how='inner')

FSList30 = np.zeros(501)
overlapList30 = np.zeros(501)
cordefList30 = np.zeros(501)
nooverlapList30 = np.zeros(501)
groupL30 = np.zeros(501)
dens30 = np.zeros(501)
JI30 = np.zeros(501)
f30 = np.zeros(501)
t30 = np.zeros(501).astype(object)
for i in np.unique(dfLinesAn30['frame']):
    dfCur30 = df_ALL30[df_ALL30['frame']==i]
    # remove all places where we look at the true that has not been matched with defined
    df_ALL30 = df_ALL30[pd.notna(df_ALL30['match index'])]
    FSList30[i] = np.median(df_ALL30['FS_coverage'][df_ALL30['frame']==i])
    
    overlapList30[i] = np.median(df_ALL30['overlap ratio'][df_ALL30['frame']==i])
    cordefList30[i] = len(np.unique(dfCur30['true index'][pd.notna(dfCur30['true index'])])) / len(np.unique(dfCur30['match index'][pd.notna(dfCur30['match index'])]))
    nooverlapList30[i] = len(dfCur30['overlap ratio'][(dfCur30['overlap ratio']==0) ]) / len(np.unique(dfCur30['true index'][pd.notna(dfCur30['true index'])])) 
    
    groupL30[i] = np.unique(df_ALL30['density groups image'][df_ALL30['frame']==i])[0]
    dens30[i] = np.unique(df_ALL30['Density full image'][df_ALL30['frame']==i])[0]
    JI30[i] = np.unique(df_ALL30['JI'][df_ALL30['frame']==i])[0]
    f30[i] = i
    t30[i] = 'line 30'

#####################
# 20 lines
df_ALLIn = dfLinesAn20.merge(df_JI20,on='frame', how='inner')
df_ALL20 = df_ALLIn.merge(df_densImage20,on='frame', how='inner')

FSList20 = np.zeros(501)
overlapList20 = np.zeros(501)
cordefList20 = np.zeros(501)
nooverlapList20 = np.zeros(501)
groupL20 = np.zeros(501)
dens20 = np.zeros(501)
JI20 = np.zeros(501)
f20 = np.zeros(501)
t20 = np.zeros(501).astype(object)
for i in np.unique(dfLinesAn20['frame']):
    dfCur20 = df_ALL20[df_ALL20['frame']==i]
    # remove all places where we look at the true that has not been matched with defined
    df_ALL20 = df_ALL20[pd.notna(df_ALL20['match index'])]
    FSList20[i] = np.median(df_ALL20['FS_coverage'][df_ALL20['frame']==i])
    
    overlapList20[i] = np.median(df_ALL20['overlap ratio'][df_ALL20['frame']==i])
    cordefList20[i] = len(np.unique(dfCur20['true index'][pd.notna(dfCur20['true index'])])) / len(np.unique(dfCur20['match index'][pd.notna(dfCur20['match index'])]))
    nooverlapList20[i] = len(dfCur20['overlap ratio'][(dfCur20['overlap ratio']==0) ]) / len(np.unique(dfCur20['true index'][pd.notna(dfCur20['true index'])])) 
    
    groupL20[i] = np.unique(df_ALL20['density groups image'][df_ALL20['frame']==i])[0]
    dens20[i] = np.unique(df_ALL20['Density full image'][df_ALL20['frame']==i])[0]
    JI20[i] = np.unique(df_ALL20['JI'][df_ALL20['frame']==i])[0]
    f20[i] = i
    t20[i] = 'line 20'
#####################
# 10 lines
df_ALLIn10 = dfLinesAn10.merge(df_JI10,on='frame', how='inner')
df_ALL10 = df_ALLIn10.merge(df_densImage10,on='frame', how='inner')

FSList10 = np.zeros(501)
overlapList10 = np.zeros(501)
cordefList10 = np.zeros(501)
nooverlapList10 = np.zeros(501)
groupL10 = np.zeros(501)
dens10 = np.zeros(501)
JI10 = np.zeros(501)
f10 = np.zeros(501)
t10 = np.zeros(501).astype(object)
for i in np.unique(dfLinesAn10['frame']):
    dfCur10 = df_ALL10[df_ALL10['frame']==i]
    # remove all places where we look at the true that has not been matched with defined
    df_ALL10 = df_ALL10[pd.notna(df_ALL10['match index'])]
    FSList10[i] = np.median(df_ALL10['FS_coverage'][df_ALL10['frame']==i])

    overlapList10[i] = np.median(df_ALL10['overlap ratio'][df_ALL10['frame']==i])
    cordefList10[i] = len(np.unique(dfCur10['true index'][pd.notna(dfCur10['true index'])])) / len(np.unique(dfCur10['match index'][pd.notna(dfCur10['match index'])]))
    nooverlapList10[i] = len(dfCur10['overlap ratio'][(dfCur10['overlap ratio']==0)  ]) / len(np.unique(dfCur10['true index'][pd.notna(dfCur10['true index'])])) 
    
    groupL10[i] = np.unique(df_ALL10['density groups image'][df_ALL10['frame']==i])[0]
    dens10[i] = np.unique(df_ALL10['Density full image'][df_ALL10['frame']==i])[0]
    JI10[i] = np.unique(df_ALL10['JI'][df_ALL10['frame']==i])[0]
    f10[i] = i
    t10[i] = 'line 10'
    
#####################
# 5 lines
df_ALLIn5 = dfLinesAn5.merge(df_JI5,on='frame', how='inner')
df_ALL5 = df_ALLIn5.merge(df_densImage5,on='frame', how='inner')

FSList5 = np.zeros(501)
overlapList5 = np.zeros(501)
cordefList5 = np.zeros(501)
nooverlapList5 = np.zeros(501)
FSList5m = np.zeros(501)
groupL5 = np.zeros(501)
dens5 = np.zeros(501)
JI5 = np.zeros(501)
f5 = np.zeros(501)
t5 = np.zeros(501).astype(object)
for i in np.unique(dfLinesAn5['frame']):
    
    dfCur5 = df_ALL5[df_ALL5['frame']==i]
    dfCur5.columns
    # remove all places where we look at the true that has not been matched with defined
    df_ALL5 = df_ALL5[pd.notna(df_ALL5['match index'])]
    FSList5[i] = np.median(df_ALL5['FS_coverage'][df_ALL5['frame']==i])
    overlapList5[i] = np.median(df_ALL5['overlap ratio'][df_ALL5['frame']==i])
    
    cordefList5[i] = len(np.unique(dfCur5['match index']))/len(np.unique(dfCur5['true index']))
    if( len(np.unique(dfCur5['true index'][pd.notna(dfCur5['true index'])])) ==0):
        nooverlapList5[i] = 1
    else:
        nooverlapList5[i] = len(dfCur5['overlap ratio'][dfCur5['overlap ratio']==0])/ len(np.unique(dfCur5['true index'][pd.notna(dfCur5['true index'])])) 
    
    FSList5m[i] = np.mean(df_ALL5['FS_coverage'][df_ALL5['frame']==i])
    
    
    groupL5[i] = np.unique(df_ALL5['density groups image'][df_ALL5['frame']==i])[0]
    dens5[i] = np.unique(df_ALL5['Density full image'][df_ALL5['frame']==i])[0]
    JI5[i] = np.unique(df_ALL5['JI'][df_ALL5['frame']==i])[0]
    f5[i] = i
    t5[i] = 'line 5'
    
#######################
# merge
   
df_ALL540 = pd.concat([df_ALL5, df_ALL10, df_ALL20, df_ALL30, df_ALL40], ignore_index=True)

FSListFull = np.concatenate((FSList40, FSList30, FSList20, FSList10, FSList5))
overlapAll = np.concatenate((overlapList40,overlapList30,overlapList20,overlapList10,overlapList5))
cordefList = np.concatenate((cordefList40,cordefList30,cordefList20,cordefList10,cordefList5))
nooverlapAll = np.concatenate((nooverlapList40,nooverlapList30,nooverlapList20,nooverlapList10,nooverlapList5))
groupFull = np.concatenate((groupL40, groupL30, groupL20, groupL10, groupL5))
densFull = np.concatenate((dens40, dens30, dens20, dens10, dens5))
JIFull = np.concatenate((JI40, JI30, JI20, JI10, JI5))
fFull = np.concatenate((f40, f30, f20, f10, f5))
tFull = np.concatenate((t40, t30, t20, t10, t5))
df_FSfull = pd.DataFrame({
    'FS coverage': FSListFull,
    'density groups image': groupFull,
    'density full image': densFull,
    'Jaccard Index': JIFull,
    'frame': fFull,
    'Line type': tFull,
    'Overlap coverage': overlapAll,
    'ratio true defined': cordefList,
    'ratio no overlap': nooverlapAll
    
    
})


# Define the conditions
conditions = [
    (df_FSfull['density groups image'] == 0),
    (df_FSfull['density groups image'] == 1),
    (df_FSfull['density groups image'] == 2),
    (df_FSfull['density groups image'] == 3),
    (df_FSfull['density groups image'] == 4),
    (df_FSfull['density groups image'] == 5),
    (df_FSfull['density groups image'] == 6),
    (df_FSfull['density groups image'] == 7),
    (df_FSfull['density groups image'] == 8),
    (df_FSfull['density groups image'] == 9)
]

# Define the corresponding values
values = ['0-.004', '.004-.006', '.006-.008','.008-.01','.01-.012','.012-.014','.014-.016','.016-.018','.018-.02','.02-1']

# Use np.select to create the new column
df_FSfull['grouping'] = np.select(conditions, values, default='object')

np.unique(df_FSfull['grouping'] )
df_FSfull.head()

df_FSfull['density groups image'].value_counts().sort_index()

df_FSfull = df_FSfull.sort_values('density groups image')

test = df_FSfull[df_FSfull['Jaccard Index']<0.1]
test['frame']
test['Line type']

np.sort(df_FSfull['Jaccard Index'].unique())

df_FSfull.columns.tolist()
df_FSfull.to_csv(overpath+'/noise001/pooled/' + 'pooled_data.csv', index=False)  



test = df_FSfull[['frame']][(df_FSfull['Jaccard Index']<0.3) & (df_FSfull['Line type']=='line 5')].values.flatten().astype(int)
test = df_FSfull[['frame']][(df_FSfull['Jaccard Index']<0.3) & (df_FSfull['Line type']=='line 10')].values.flatten().astype(int)

###############################################################################
#
# plot pooled data
#
###############################################################################

figsSave = overpath+'paper_figs/'

plt.close('all')

fig, axs = plt.subplot_mosaic("AB;CD",  figsize=(8.27,6))

sns.lineplot(data=df_FSfull, x='grouping', y='FS coverage',estimator='mean', ci=90,label='GraFT',ax=axs['A']) #for 95% confidence interval
axs['A'].tick_params(axis='x', labelrotation=30)
axs['A'].legend(frameon=False)

sns.lineplot(data=df_FSfull, x='grouping', y='Overlap coverage',estimator='mean',  ci=90,label='GraFT',ax=axs['B']) #for 95% confidence interval
axs['B'].tick_params(axis='x', labelrotation=30)
axs['B'].legend().remove()

sns.lineplot(data=df_FSfull, x='grouping', y='ratio no overlap',estimator='mean', ci=90,label='GraFT',ax=axs['C'])
axs['C'].tick_params(axis='x', labelrotation=30)
axs['C'].legend().remove()

sns.lineplot(data=df_FSfull, x='grouping', y='Jaccard Index',estimator='mean', ci=90,label='GraFT',ax=axs['D']) #for 95% confidence interval
plt.ylim(0.65,1)
axs['D'].tick_params(axis='x', labelrotation=30)
axs['D'].legend().remove()

axs['A'].set_ylim(0,1)
axs['B'].sharey(axs['A'])
axs['C'].sharey(axs['A'])
axs['D'].sharey(axs['A'])

axs['A'].set_xlabel(None)
axs['B'].set_xlabel(None)
axs['C'].set_xlabel("Density image")
axs['D'].set_xlabel("Density image")

axs['A'].set_ylabel("Filament matched coverage")
axs['B'].set_ylabel("Filament coverage")
axs['C'].set_ylabel("No coverage to true filament")
axs['D'].set_ylabel("Jaccard Index")


axs['A'].tick_params(labelbottom=False)
axs['B'].tick_params(labelbottom=False)


for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')
    

plt.tight_layout()

plt.savefig(figsSave+'graft_performance.pdf')
