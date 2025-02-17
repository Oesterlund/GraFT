#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: isabella
"""

import collections
import numpy as np
import pandas as pd
overpath = 'define_this_path'

track_all = np.zeros((10,20))

for dm in range(10):
        
    df_Ori = pd.read_csv(overpath+'/others_code/TSOAX/timeseries/10_lines_stack/density/0{0}_10_lines.csv'.format(dm))
    
    df_Ori.columns
    
    uframe = np.unique(df_Ori['frame'])
    trackList = np.zeros(len(uframe))
    
    
    dflInesI = df_Ori[~np.isnan(df_Ori['best_match_line1_id'])]
    overlapList = np.zeros(len(uframe))
    
    for n in range(len(uframe)):
        
        dflInesICov = dflInesI[dflInesI['frame']== n].copy()
        dflInesICov[['best_IDmatch_line1_id','overlap ratio']]
            
        dflInesICov['FS_coverage'] = dflInesICov['overlap_count']#/dflInesICov['true_len']
        
        dflInesICov['FS_coverage'] = dflInesICov['FS_coverage'].fillna(0)
        
        overlapList[n] = np.median(dflInesICov['overlap ratio'])
        
        
    
        # for the first frame we need to save the correct matches
        if(n==0):
            matchedVals=[]
            matchVal = dflInesICov['best_IDmatch_line1_id'].dropna()    
            for kl,val in zip(range(len(matchVal)),matchVal):
                vlas = dflInesICov[['best_IDmatch_line1_id','overlap ratio']][(dflInesICov['best_IDmatch_line1_id'] == val)]
                if(vlas['overlap ratio']>0.8).all():
                    matchedVals.append(vlas['best_IDmatch_line1_id'].item())
            trackList[n] = len(matchedVals)/5
                    
        else:
            if(len(matchedVals)!=5):
                
                matchedValsInter=[]
                matchValInter = dflInesICov['best_IDmatch_line1_id'].dropna()    
                for kl,val in zip(range(len(matchValInter)),matchValInter):
                    vlas = dflInesICov[['best_IDmatch_line1_id','overlap ratio']][(dflInesICov['best_IDmatch_line1_id'] == val)]
                    if(vlas['overlap ratio']>0.8).all():
                        matchedValsInter.append(vlas['best_IDmatch_line1_id'].item())
                
                # search for the new values appearing
                mvi = set(matchedVals + matchedValsInter)
                if(len(mvi)>=5):
                    matchedVals=list(mvi)
                else:
                    print('error', n)
                    
            result = np.zeros(len(matchedVals))
            for kl,val in zip(range(len(matchedVals)),matchedVals):
                # Apply the condition: 'best_IDmatch_line1_id' equals the current value and 'overlap ratio' > 0.8
                if(dflInesICov['best_IDmatch_line1_id'] == val).any():
                    result[kl] = float(dflInesICov['overlap ratio'][(dflInesICov['best_IDmatch_line1_id'] == val)].iloc[0])
                else:
                    result[kl] =0
         
            trackList[n] = (result[(result >0.8)]).size/5
            
    track_all[dm] = trackList

np.save(overpath+'/others_code/TSOAX/timeseries/10_lines_stack/TSOAX_lines_track.csv', track_all)  

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(track_all[i], color='blue', alpha=0.1)  # Plot each array with alpha=0.3

mean_array = track_all.mean(axis=0)

# Plot the mean array with alpha=1
plt.plot(mean_array, color='red', alpha=1, label='Mean')

plt.savefig(overpath+'/others_code/TSOAX/timeseries/10_lines_stack/figs/TSOAX_performance.png')