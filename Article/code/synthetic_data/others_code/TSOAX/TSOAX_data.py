#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:44:58 2024

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.colors import ListedColormap
#import skimage.io as io
#import scienceplots
import tifffile
import seaborn as sns
import os
import re
from scipy.optimize import linear_sum_assignment
import math

#plt.style.use(['science','nature']) # sans-serif font
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



###############################################################################
#
# functions
#
###############################################################################

def calcDiff(img_o,imgSnake):
    #must add padding
    img_o = np.pad(img_o,3, 'constant')
    imgSnake = np.pad(imgSnake,3, 'constant')
    mask=imgSnake.copy()
    # the original skeleton version of image / true location
    imgOri = (img_o>0)*1
    
    count = 0
    #perfect overlap search
    (rows,cols) = np.nonzero((mask>0)*1)
    for k in range(len(rows)):
        r = rows[k]
        c = cols[k]
        if(imgOri[r,c]==1):
            imgOri[r,c] = 0
            mask[r,c] = 0
            count += 1
    
     # 1 radii search
    (rows,cols) = np.nonzero((mask>0)*1)
    for j in range(len(rows)):
        r = rows[j]
        c = cols[j]
        
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        col_neighOri = col_neigh.astype('int')
        row_neighOri = row_neigh.astype('int')
        imageSection = imgOri[row_neighOri,col_neighOri]
        if(np.max(imageSection)>0):
            ind = np.where(imageSection)
            index = 0
            if(len(ind[0])!=1):
                index = int(np.floor(len(ind)/2.))
            r2 = (r+ind[0])[index]-1
            c2 = (c+ind[1])[index]-1
            
            imgOri[r2,c2] = 0
            mask[r,c] = 0
            count += 1
    
     # 2 radii search
    (rows,cols) = np.nonzero((mask>0)*1)
    for l in range(len(rows)):
        r = rows[l]
        c = cols[l]
        
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-2,c-1,c,c+1,c+2]), np.array([r-2,r-1,r,r-1,r+2]))
        col_neighOri = col_neigh.astype('int')
        row_neighOri = row_neigh.astype('int')
        imageSection = imgOri[row_neighOri,col_neighOri]
        if(np.max(imageSection)>0):
            ind = np.where(imageSection)
            index = 0
            if(len(ind[0])!=1):
                index = int(np.floor(len(ind)/2.))
            r3 = (r+ind[0])[index]-1
            c3 = (c+ind[1])[index]-1
            
            imgOri[r3,c3] = 0
            mask[r,c] = 0
            count += 1
     
    # 3 radii search
    (rows,cols) = np.nonzero((mask>0)*1)
    for mn in range(len(rows)):
        r = rows[mn]
        c = cols[mn]
        
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-3,c-2,c-1,c,c+1,c+2,c+3]), np.array([r-3,r-2,r-1,r,r-1,r+2,r+3]))
        col_neighOri = col_neigh.astype('int')
        row_neighOri = row_neigh.astype('int')
        imageSection = imgOri[row_neighOri,col_neighOri]
        if(np.max(imageSection)>0):
            ind = np.where(imageSection)
            index = 0
            if(len(ind[0])!=1):
                index = int(np.floor(len(ind)/2.))
            r4 = (r+ind[0])[index]-1
            c4 = (c+ind[1])[index]-1
            
            imgOri[r4,c4] = 0
            mask[r,c] = 0
            count += 1
            
    FP = np.sum(mask)
    correctC = np.sum((img_o>0)*1)        
    perC = count/correctC*100
    FN = (correctC - count)/correctC*100
    FP_trueimg = FP/correctC*100
    countMask = np.sum(imgSnake)
    '''
    plt.figure()
    plt.imshow(mask)
    '''
    return correctC, count,countMask, perC,FP_trueimg, FN

###############################################################################
#
# load in txt files from TSOAX, and reformat to pandas dataframe for datapart
#
###############################################################################

def JI(TP,FP,FN):
    JI = TP/(TP + FP + FN)
    return JI


def points_within_radius(x1, y1, x2, y2, radius=3):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= radius

# Function to check if two lines overlap within a certain radius
def lines_overlap(line1, line2, radius=3):
    line1 = line1.reset_index(drop=True)
    line2 = line2.reset_index(drop=True)
    overlap_count = 0
    used_points_line1 = np.zeros(len(line1), dtype=bool)
    used_points_line2 = np.zeros(len(line2), dtype=bool)
    
    for idx1, row1 in line1.iterrows():
        for idx2, row2 in line2.iterrows():
            if not used_points_line1[idx1] and not used_points_line2[idx2]:
                if points_within_radius(row1['x'], row1['y'], row2['x_positions'], row2['y_positions']):
                    overlap_count += 1
                    used_points_line1[idx1] = True
                    used_points_line2[idx2] = True
                    break  # Move to the next point in line1 once a match is found
    if(overlap_count != 0):
        overlap_count = overlap_count/len(line2)
    return overlap_count

def identification_LAP(pathNonoise,snakesXY,savepathLines, box_size, linetype, frame):
    df_Ori = pd.read_csv(pathNonoise)
    
    # filter away all values outside our frame of view
    df_Ori = df_Ori[(df_Ori['x_posR'] >= 0) & (df_Ori['x_posR'] < box_size)]
    df_Ori = df_Ori[(df_Ori['y_posR'] >= 0) & (df_Ori['y_posR'] < box_size)]

    df_timeseries = df_Ori[df_Ori['frame']==frame].copy()
    
    df_snakes = snakesXY.copy()
    # Remove duplicate points based on the rounded x and y values
    df_snakes = df_snakes.drop_duplicates(subset=['x', 'y'])
            
    results = []
    lines_idTest = df_snakes['snake index'].unique()
    trueS = len(df_timeseries['line_no'].unique())
    defineS = len(df_snakes['snake index'].unique())
    costM = np.zeros((trueS,defineS))
    trueSLst = df_timeseries['line_no'].unique()
    
    for line2_id,m in zip(trueSLst,range(trueS)):
        line2 = df_timeseries[df_timeseries['line_no'] == line2_id]

        for line1_id,k in zip(lines_idTest,range(defineS)):
            line1 = df_snakes[df_snakes['snake index'] == line1_id]
            costM[m,k] = lines_overlap(line1, line2)
    
    row_ind, col_ind = linear_sum_assignment(costM,maximize=True)
    
    definedM = np.zeros(len(row_ind))
    deleteDef = []
    for n in range(len(row_ind)):
        lenLine2 = len(df_timeseries[df_timeseries['line_no'] == trueSLst[n]])
        lenLine1 = len( df_snakes[df_snakes['snake index'] == lines_idTest[col_ind[n]]])
        definedM[n] = lines_idTest[col_ind[n]]
        if(costM[row_ind[n],col_ind[n]]==0):
            deleteDef = np.append(deleteDef,int(n))
            bestM = None
            lenLine1 = 0
        else:
            bestM = definedM[n]
                        
        results.append({'line2_id': trueSLst[n], 'best_match_line1_id': bestM, 'true_len':lenLine2, 'found_len':lenLine1, 'overlap_count': costM[row_ind[n],col_ind[n]],'overlap ratio': costM[row_ind[n],col_ind[n]]})
    if(len(deleteDef)!=0):
        definedM = np.delete(definedM,deleteDef.astype(int))
    main_list = list(set(lines_idTest) - set(definedM))
    for x in range(len(main_list)):
        lenLine2 = None
        lenLine1 = len( df_snakes[df_snakes['snake index'] == main_list[x]])
        # find out if this snake oerlapped a true filament
        indices = np.where(lines_idTest == main_list[x])
        overlap = np.max(costM[:,indices])
        results.append({'line2_id': None, 'best_match_line1_id': main_list[x], 'true_len':0, 'found_len':lenLine1, 'overlap_count': 0,'overlap ratio': overlap})

    # Convert results to a dataframe for better visualization
    results_df = pd.DataFrame(results)
        
    results_df.to_csv(savepathLines + '{0}_{1}_df_line_comparison.csv'.format(linetype, frame), index=False)  
    return 

def identification_LAP_time(pathNonoise,snakesXY, box_size):
    df_Ori = pd.read_csv(pathNonoise)
    # filter away all values outside our frame of view
    df_Ori = df_Ori[(df_Ori['x_posR'] >= 0) & (df_Ori['x_posR'] < box_size)]
    df_Ori = df_Ori[(df_Ori['y_posR'] >= 0) & (df_Ori['y_posR'] < box_size)]

    
    
    df_snakes = snakesXY.copy()
    # Remove duplicate points based on the rounded x and y values
    df_snakes = df_snakes.drop_duplicates(subset=['x', 'y'])
            
    results = []
    
    for k in np.unique(df_snakes['frame']):
        print(k)
 
        df_timeseries = df_Ori[df_Ori['frame']==k].copy()
        df_snakes_ts = df_snakes[df_snakes['frame']==k].copy()
            
        lines_idTest = df_snakes_ts['snake index'].unique()
        lines_idmatch = df_snakes_ts.groupby('snake index')['match_id'].unique().tolist()
        lines_idmatch = [x[0] if len(x) > 0 else np.nan for x in lines_idmatch]
        trueS = len(df_timeseries['line_no'].unique())
        defineS = len(df_snakes_ts['snake index'].unique())
        costM = np.zeros((trueS,defineS))
        trueSLst = df_timeseries['line_no'].unique()
        
        for line2_id,m in zip(trueSLst,range(trueS)):
            line2 = df_timeseries[df_timeseries['line_no'] == line2_id]
    
            for line1_id,ik in zip(lines_idTest,range(defineS)):
                line1 = df_snakes_ts[df_snakes_ts['snake index'] == line1_id]
                costM[m,ik] = lines_overlap(line1, line2)
        
        row_ind, col_ind = linear_sum_assignment(costM,maximize=True)
        
        definedM = np.zeros(len(row_ind))
        definedID = np.zeros(len(row_ind))
        deleteDef = []
        for n in range(len(row_ind)):
            lenLine2 = len(df_timeseries[df_timeseries['line_no'] == trueSLst[n]])
            lenLine1 = len(df_snakes_ts[df_snakes_ts['snake index'] == lines_idTest[col_ind[n]]])
            definedM[n] = lines_idTest[col_ind[n]]
            definedID[n] = lines_idmatch[col_ind[n]]
            if(costM[row_ind[n],col_ind[n]]==0):
                deleteDef = np.append(deleteDef,int(n))
                bestM = None
                lenLine1 = 0
            else:
                bestM = definedM[n]
                bestID = definedID[n]
                            
            results.append({'frame':k, 'line2_id': trueSLst[n], 'best_match_line1_id': bestM, 'best_IDmatch_line1_id': bestID, 'true_len':lenLine2, 'found_len':lenLine1, 'overlap_count': costM[row_ind[n],col_ind[n]],'overlap ratio': costM[row_ind[n],col_ind[n]]})
        if(len(deleteDef)!=0):
            definedM = np.delete(definedM,deleteDef.astype(int))
        main_list = list(set(lines_idTest) - set(definedM))
        # use the index of lines_idTest from main list to get correct values
        main_IDlist = [lines_idmatch[np.where(lines_idTest == value)[0][0]] if value in lines_idTest else None for value in main_list]

        for x in range(len(main_list)):
            lenLine2 = None
            lenLine1 = len( df_snakes_ts[df_snakes_ts['snake index'] == main_list[x]])
            # find out if this snake oerlapped a true filament
            indices = np.where(lines_idTest == main_list[x])
            overlap = np.max(costM[:,indices])
            results.append({'frame':k, 'line2_id': None, 'best_match_line1_id': main_list[x],'best_IDmatch_line1_id': main_IDlist[x], 'true_len':0, 'found_len':lenLine1, 'overlap_count': 0,'overlap ratio': overlap})

    # Convert results to a dataframe for better visualization
    results_df = pd.DataFrame(results)

    return results_df


def identification(pathNonoise,snakesXY,savepathLines, box_size, linetype, frame):
    df_Ori = pd.read_csv(pathNonoise)
    
    # filter away all values outside our frame of view
    df_Ori = df_Ori[(df_Ori['x_posR'] >= 0) & (df_Ori['x_posR'] < box_size)]
    df_Ori = df_Ori[(df_Ori['y_posR'] >= 0) & (df_Ori['y_posR'] < box_size)]

    df_timeseries = df_Ori[df_Ori['frame']==frame].copy()
    
    df_snakes = snakesXY.copy()
    # Remove duplicate points based on the rounded x and y values
    df_snakes = df_snakes.drop_duplicates(subset=['x', 'y'])
            
    results = []
    line1S = []
    lines_idTest = df_snakes['snake index'].unique()
    
    for line2_id in df_timeseries['line_no'].unique():
        line2 = df_timeseries[df_timeseries['line_no'] == line2_id]
        max_overlap = 0
        best_match_line1_id = None
        line1L = 0
        for line1_id in lines_idTest:
            line1 = df_snakes[df_snakes['snake index'] == line1_id]
            overlap = lines_overlap(line1, line2)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_line1_id = line1_id
                line1L = len(line1)
        if(best_match_line1_id!=None):
            lines_idTest = [item for item in lines_idTest if item != best_match_line1_id]
            line1S = np.append(line1S,best_match_line1_id)
        elif():
            line1L = 0
        results.append({'line2_id': line2_id, 'best_match_line1_id': best_match_line1_id, 'true_len':len(line2), 'found_len':line1L, 'overlap_count': max_overlap})
    if len(lines_idTest)!=0:
        for s in lines_idTest:
            line1 = df_snakes[df_snakes['snake index'] == s]
            results.append({'line2_id': None, 'best_match_line1_id': s, 'true_len':0, 'found_len':len(line1), 'overlap_count': 0})
            
    # Convert results to a dataframe for better visualization
    results_df = pd.DataFrame(results)
        
    results_df.to_csv(savepathLines + '{0}_{1}_df_line_comparison.csv'.format(linetype, frame), index=False)  
    return 


def snakes(path_snake,linetype, frame,frame_graft,savepath,imgpath,pathNonoise,box_size):
    #path_snake = '/Users/pgf840/Documents/PLEN/tracking-filaments/articles_code/TSOAX/TSOAX_noise_test/noise/noise/snakes_2.txt'
    col_names = ['snake index', 'point index', 'x','y','z','int']
    snakes = pd.DataFrame(columns = col_names)
    values = []
    switch=0
    with open(path_snake) as f:
    #with open('/Users/pgf840/Documents/PLEN/tracking-filaments/articles_code/TSOAX/TSOAX_noise_test/noise/noise/snakes_2.txt') as f:
    #with open('/Users/pgf840/Documents/PLEN/tracking-filaments/articles_code/TSOAX/TSOAX_noise_test/blur/blur/snakes_0.txt') as f:
        for line in f:
            
            if((switch==1) & (line.startswith('#')==False) & (line.startswith('[')==False)):
                
                values=[float(i) for i in line[0:-1].split()]
                snakes.loc[-1]=values
                # update dataframe index
                snakes.index=snakes.index+1
                snakes = snakes.sort_index() 
                
                
            elif(line.startswith('#')): #(line=='#1\n'):
                #first line needs special treatment
                switch=1
            else:
                switch=0
    
    np.unique(snakes['snake index'])
    print('filaments deteceted: ',len(np.unique(snakes['snake index'])))
    ###############################################################################
    #
    # compare to npy true image files
    #
    ###############################################################################
   
    with tifffile.TiffFile(imgpath) as tif:
        img_o = tif.asarray()
    
    imgTrue = img_o
    
    snakesXY = snakes[['snake index','x','y']]
    snakesXY = snakesXY.rename(columns={"y": "y in"})
    
    #turn to be same as true image
    snakesXY['y'] = abs(snakesXY['y in']-500)
    snakesXY = snakesXY.drop(['y in'], axis=1)
    
    snakesXY['snake index'].unique()
    snakesXY['x'] = snakesXY['x'].round().astype(int)
    snakesXY['y'] = snakesXY['y'].round().astype(int)
    
    # Remove duplicate points based on the rounded x and y values
    snakesXY = snakesXY.drop_duplicates(subset=['x', 'y'])
    
    identification_LAP(pathNonoise,snakesXY,savepath,box_size, linetype, frame_graft)
    
    indexS = np.unique(snakesXY['snake index'])
    valImg = np.arange(1,len(indexS)+1)
    
    imgSnakes = np.zeros((500, 500))
    imgSnakesM = np.zeros((500, 500))
    for i in range(len(snakesXY)):
        
        index,x,y=snakesXY.iloc[i]
        ind = np.where(indexS==index)[0][0]
        if((int(x)<500) and (int(y)<500)):
            imgSnakes[int(y),int(x)]=valImg[ind]
            imgSnakesM[int(y),int(x)] = 1
        
    plt.figure(figsize=(5,5))
    plt.imshow(imgSnakesM)
    #plt.colorbar()
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('/home/isabella/Documents/PLEN/dfs/others_code/TSOAX/figs/img_{0}_{1}'.format(linetype, frame))
    plt.close('all')
    
    '''
    
    #cmap = colors.ListedColormap(['white', 'red','blue','green','cyan','yellow','orange','brown','purple','lime','navy','hotpink','maroon','violet','gray','olive','aqua','turquoise','black'])

    snakes_int = snakes.copy()
    
    snakes_int['y'] = abs(snakes_int['y']-500)
    
    snakesXY = snakes_int[['snake index','x','y']]
    
    #turn to be same as true image
    #snakesXY['y new'] = abs(snakes_int['y']-500)
    
    indexS = np.unique(snakes_int['snake index'])
    valImg = np.arange(1,len(indexS)+1)
    
    imgSnakes = np.zeros((500, 500), dtype=np.double)
    imgSnakesM = np.zeros((500, 500), dtype=np.double)
    for i in range(len(snakesXY)):
        
        index,x,y=snakesXY.iloc[i]
        ind = np.where(indexS==index)
        
        if((int(x)<500) and (int(y)<500)):
            imgSnakes[int(y),int(x)]=valImg[ind]
            imgSnakesM[int(y),int(x)] = 1
    plt.figure(figsize=(5,5))
    plt.imshow(imgSnakes)
    #plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(savepath+'img{0}.png'.format(value), format='png', dpi=350)
    plt.close('all')

    import scipy
    test=scipy.ndimage.zoom(imgSnakes, 0.7,order=5)
    
    plt.figure()
    plt.imshow(test.astype(int),cmap=cmap)
    '''
    return 

def snakes_time(path_snake,savepath, savefile, images_path,pathNonoise,track_len, filename,box_size):
    col_names = ['snake index', 'point index', 'x','y','z','int','frame']
    snakes = pd.DataFrame(columns = col_names)
    values = []
    switch=0
    count=0
    track_start=0
    track_names = []
    for nm in range(track_len):
        track_names.append('tracks {0}'.format(nm))
    tracks = pd.DataFrame(columns = track_names)
    track_key = []
    keep_lines= False
    with open(path_snake) as f:
        
        for line in f:

            if(line.startswith('$')==True):
                count += 1
                print('count',count)
                
            elif((switch==1) & (line.startswith('#')==False) & (line.startswith('$')==False) & (line.startswith('[')==False) & (line.startswith('T')==False)):
                
                values=[float(i) for i in line[0:-1].split()]
                values.append(count)
                snakes.loc[-1]=values
                # update dataframe index
                snakes.index=snakes.index+1
                snakes = snakes.sort_index() 
                
                
            elif(line.startswith('#')): #(line=='#1\n'):
                #first line needs special treatment
                switch=1
            else:
                if(line.startswith('T')==True):
                    switch=0
                    track_start=1
            if((track_start==1) & (line.startswith('T')==False)):
                    
                    values=[float(i) for i in line[0:-1].split()]
                    tracks.loc[-1]=values
                    tracks.index=tracks.index+1
                    tracks = tracks.sort_index() 
                    
            if keep_lines:
                values = [int(x) for x in line.split()]
                track_key.append(np.array(values))
            if 'Tracks' in line:
                keep_lines = True  # Start keeping lines when 'Tracks' is found
                
    track_key = np.array(track_key)
    
    snakes['match_id'] = np.nan

    # Iterate through each array in 'track_key' and assign a unique number based on matches with 'snake index'
    unique_id_counter = 0
    
    for i, track_array in enumerate(track_key):

        # Check for matching values between the 'snake index' and current 'track_key' array
        mask = snakes['snake index'].isin(track_array)
        
        # Assign a unique number to the new column for rows where 'snake index' matches
        snakes.loc[mask, 'match_id'] = unique_id_counter
        
        # Increment the unique ID counter for the next set of matches
        unique_id_counter += 1
    
    
    np.unique(snakes['snake index'])
    
    print('filaments deteceted: ',len(np.unique(snakes['snake index'])))
    
    #### tracing
    snakesXY_time = snakes[['frame','snake index','x','y','match_id']]
    snakesXY_time.loc[:, 'y'] = abs(snakesXY_time['y'] - box_size)
    df_id_time = identification_LAP_time(pathNonoise,snakesXY_time, box_size)
    df_id_time.to_csv(savefile + filename + '.csv', index=False)  

    ###############################################################################
    #
    # compare to npy true image files
    #
    ###############################################################################
   
    #cmap = colors.ListedColormap(['white','blue','green','cyan','yellow','orange','brown','purple','lime','navy','hotpink','maroon','violet','gray','olive','aqua','turquoise','black','red'])
    
    color_mapping = {
    0: 'white',
    1: 'hotpink',
    2: 'green',
    3: 'blue',
    4: 'orange',
    5: 'purple',
    6: 'turquoise',
    7: 'red'
    }
    
    colors = [color_mapping[i] for i in sorted(color_mapping.keys())]
    cmap = ListedColormap(colors)

    with tifffile.TiffFile(images_path) as tif:
        img_o = tif.asarray()
    
    for d in range(len(img_o)):
        # create colorlist per match
        

        snakes_int = snakes[snakes['frame']==d].copy()
        
        #snakes_int.loc[:, 'y'] = abs(snakes_int['y'] - 500)
        snakes_int['y'] = abs(snakes_int['y']-500)
        
        snakesXY = snakes_int[['snake index','x','y','match_id']]
    
        
        #turn to be same as true image
        #snakesXY['y new'] = abs(snakes_int['y']-500)
        indexS = np.unique(snakes_int['snake index'])
        print(d,indexS)
        valImg = np.arange(1,len(indexS)+1)
        
        imgSnakes = np.zeros((500, 500), dtype=np.double)
        imgSnakesM = np.zeros((500, 500), dtype=np.double)
        
        for i in range(len(snakesXY)):
            
            index,x,y,match=snakesXY.iloc[i]
            
            if isinstance(match, (int, float)) and not math.isnan(match):
                
                imgSnakes[int(y)-1,int(x)-1]= match + 1
                imgSnakesM[int(y)-1,int(x)-1] = 1
            else:
                imgSnakes[int(y)-1,int(x)-1]= unique_id_counter+1
                imgSnakesM[int(y)-1,int(x)-1] = 1
                
        '''
        for i in range(len(snakesXY)):
            
            index,x,y=snakesXY.iloc[i]
            ind = np.where(indexS==index)
            if(index in np.asarray(tracks['tracks {0}'.format(d)])):
                col_ind = np.where(tracks['tracks {0}'.format(d)] ==index)[0][0]+1
                
                #print(col_ind)
            else:
                # remove the ones that are not tracked by setting their value to 0
                # otherwise always have them black by setting value equal to 18
                col_ind= 18
            imgSnakes[int(y)-1,int(x)-1]=valImg[ind] # col_ind
            imgSnakesM[int(y)-1,int(x)-1] = 1
        '''
     
           
        plt.figure(figsize=(5,5))
        plt.imshow(imgSnakes,cmap=cmap, vmin=0, vmax=7)
        plt.colorbar(ticks=np.arange(0,8))
        plt.tight_layout()
        plt.savefig(savepath+'img{0}.png'.format(d), format='svg', dpi=1200)
        plt.close('all')
        #print('true count,count of overlap, image count,TP, FP, FN: ', calcDiff(imgTrue,imgSnakesM))
    return

def JI_loop(imgpath, path_snake):
    
    col_names = ['snake index', 'point index', 'x','y','z','int']
    snakes = pd.DataFrame(columns = col_names)
    values = []
    switch=0
    with open(path_snake) as f:
    #with open('/Users/pgf840/Documents/PLEN/tracking-filaments/articles_code/TSOAX/TSOAX_noise_test/noise/noise/snakes_2.txt') as f:
    #with open('/Users/pgf840/Documents/PLEN/tracking-filaments/articles_code/TSOAX/TSOAX_noise_test/blur/blur/snakes_0.txt') as f:
        for line in f:
            
            if((switch==1) & (line.startswith('#')==False) & (line.startswith('[')==False)):
                
                values=[float(i) for i in line[0:-1].split()]
                snakes.loc[-1]=values
                # update dataframe index
                snakes.index=snakes.index+1
                snakes = snakes.sort_index() 
                
                
            elif(line.startswith('#')): #(line=='#1\n'):
                #first line needs special treatment
                switch=1
            else:
                switch=0
    
    snakesXY = snakes[['snake index','x','y']]
    snakesXY = snakesXY.rename(columns={"y": "y in"})
    
    #turn to be same as true image
    snakesXY['y'] = abs(snakesXY['y in']-500)
    snakesXY = snakesXY.drop(['y in'], axis=1)
    
    snakesXY['snake index'].unique()
    snakesXY['x'] = snakesXY['x'].round().astype(int)
    snakesXY['y'] = snakesXY['y'].round().astype(int)
    
    # Remove duplicate points based on the rounded x and y values
    snakesXY = snakesXY.drop_duplicates(subset=['x', 'y'])
    
    indexS = np.unique(snakesXY['snake index'])
    valImg = np.arange(1,len(indexS)+1)
    imgSnakes = np.zeros((500, 500))
    imgSnakesM = np.zeros((500, 500))
    for i in range(len(snakesXY)):
        
        index,x,y=snakesXY.iloc[i]
        ind = np.where(indexS==index)[0][0]
        if((int(x)<500) and (int(y)<500)):
            imgSnakes[int(y),int(x)]=valImg[ind]
            imgSnakesM[int(y),int(x)] = 1
    
    with tifffile.TiffFile(imgpath) as tif:
        img_o = tif.asarray()
    
    imgTrue = img_o
    correctCount,countIm,count,TPI,FPI,FNI = calcDiff(imgTrue,imgSnakesM)
    print('true count,count of overlap, image count,TP, FP, FN: ', correctCount,countIm,count,TPI,FPI,FNI )
    
    
    JI_val = JI(TPI,FPI,FNI)
    return JI_val
