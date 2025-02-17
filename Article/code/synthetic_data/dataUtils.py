#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:28:14 2024

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from skimage.draw import bezier_curve
from skimage.draw import disk
import tifffile
import re
import networkx as nx
import itertools
import sys
import pickle
from scipy.optimize import linear_sum_assignment
#path="/Users/pgf840/Documents/PLEN/tracking-filaments/dfs/old_utils/"

path="/home/isabella/Documents/PLEN/dfs/data/Zengyu/"
sys.path.append(path)

import utilsF
            
'''
###############################################################################

Function to create a synthetic dataset representing filamentous structures
This function creates lines represetned by bezier curve, and adds curvature on it
dependent on the length of the lines

###############################################################################
'''

def generate_filamentous_structure(box_size, num_lines, curvature_factor):
    
    '''
    Parameters
    ----------
    box_size : int
        size of the 2D bounding box the lines will be in.
    num_lines : int
        Number of lines generated, these lines will excist in all frames.
    curvature_factor : int
        A factor for curvatore given to the bezel function governing the line curvature.
        This factor is affected by a random variable that is dependent on length.
    Returns
    -------
    df_data : DataFrame
        A dataframe containing information of the timeseries data generated.
        It contains x,y coordinates of the lines for each frame, the intensity and length of the line.

    '''
    plt.close('all')
    # Initial setup for plots and data storage
    #plt.figure(figsize=(10, 10))
    df_data = pd.DataFrame()

    max_possible_length = box_size #np.sqrt(2) * box_size

    # Generate initial lines and their properties
    lines_properties = []
    line_lengthL = np.zeros(num_lines)
    for i in range(num_lines):
        
        # make relative shorter filaments more likely
        # perhaps use power law
        line_length = np.random.exponential(scale=max_possible_length * 0.3)
        # cut the val off if its longer than max length
        line_length = min(line_length, max_possible_length)
        line_length = max(25, line_length)
        
        normalized_influence = (line_length - 25) / (box_size - 25)
        line_intensity = 0.5 + (normalized_influence * 0.5) * np.random.random()
        
        #line_intensity = min((line_length * (1 + np.random.uniform(0., 0.8))), max_possible_length) / max_possible_length
        # random weight to be added to beizer control points. This value should max go to line_intensity
        random_weight = np.max([line_intensity - np.random.uniform(0, line_intensity), 0.2])
        curvature_weight = curvature_factor * (1 - random_weight)
        
        # Initial angle biased towards the major axis
        # if x, normal around 0, with scale pi/4, else normal around pi/2 (90) 
        angle = np.random.uniform(0,np.pi) # 0-180
        '''
        if major_axis == 'x':
            angle = np.random.normal(0, np.pi / 4 ) * axis_weight*random_weight
        else:
            angle = np.random.normal(np.pi / 2, np.pi / 4) * axis_weight*random_weight
        '''
            
        lines_properties.append({
            'line_length': line_length,
            'line_intensity': line_intensity,
            'angle': angle,
            'curvature': curvature_weight
        })
        line_lengthL[i] = line_length 
    iL = np.argmax(line_lengthL)
    item = lines_properties[iL]
    lines_properties.pop(iL)
    lines_properties.insert(0,item)
    # Place in a bounding box
    for i, line in enumerate(lines_properties):
        # generate the beizer curve

        if(i==0):
            start_x, start_y = np.random.randint(0, box_size/2, 2)
            
            end_x = start_x + line['line_length'] * np.cos(line['angle'])
            end_y = start_y + line['line_length'] * np.sin(line['angle'])
            
            #check if first line is inside the box
            while((0>end_x) & (end_x>box_size) & (0>end_y) & (end_y>box_size)):
                start_x, start_y = np.random.randint(0, box_size, 2)
                
                end_x = start_x + line['line_length'] * np.cos(line['angle'])
                end_y = start_y + line['line_length'] * np.sin(line['angle'])
                
            x_or_y = np.random.randint(0,2)
            if(x_or_y==0):
                control_x = (start_x + end_x) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                control_y = (start_y + end_y) / 2 + line['line_length']*0.05
                if((start_x<control_x) & (end_x<control_x)):
                    control_x=end_x-0.5
                elif((start_x > control_x) & (end_x > control_x)):
                    control_x=start_x-0.5
                if((start_y < control_y) & (end_y < control_y)):
                    control_y=end_y-0.5
                elif((start_y > control_y) & (end_y>control_y)):
                    control_y = start_y-0.5
                    
                
            else:
                control_x = (start_x + end_x) / 2  + line['line_length']*0.05
                control_y = (start_y + end_y) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                if((start_y < control_y) & (end_y < control_y)):
                    control_y=end_y-0.5
                elif((start_y > control_y) & (end_y>control_y)):
                    control_y = start_y-0.5
                if((start_x<control_x) & (end_x<control_x)):
                    control_x=end_x-0.5
                elif((start_x > control_x) & (end_x > control_x)):
                    control_x=start_x-0.5

            curve_x,curve_y = bezier_curve(np.round(start_x).astype(int), np.round(start_y).astype(int), 
                                           np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                           np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
            #plt.plot(curve_x, curve_y, alpha=0.7)
            #plt.xlim(0, box_size)
            #plt.ylim(0, box_size)
            
            df_linTemp = pd.DataFrame({
                'line_no': i,
                'line_intensity': line['line_intensity'],
                'x_positions': curve_x,
                'y_positions': curve_y
            })
            
            df_data = pd.concat([df_data, df_linTemp], ignore_index=True)
        else:
            #create small probability of a filament lying a random space
            random_number = np.random.beta(normalized_influence * 5 + 1, (1 - normalized_influence) * 5 + 1)
    
            if(random_number>=0.75):
                start_x, start_y = np.random.randint(0, box_size/2, 2)
                
                end_x = start_x + line['line_length'] * np.cos(line['angle'])
                end_y = start_y + line['line_length'] * np.sin(line['angle'])
                
                #check if first line is inside the box
                while((0>end_x) & (end_x>500) & (0>end_y) & (end_y>500)):
                    start_x, start_y = np.random.randint(0, box_size, 2)
                    
                    end_x = start_x + line['line_length'] * np.cos(line['angle'])
                    end_y = start_y + line['line_length'] * np.sin(line['angle'])
                    
                x_or_y = np.random.randint(0,2)
                if(x_or_y==0):
                    control_x = (start_x + end_x) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                    control_y = (start_y + end_y) / 2 + line['line_length']*0.05
                    if((start_x<control_x) & (end_x<control_x)):
                        control_x=end_x-0.5
                    elif((start_x > control_x) & (end_x > control_x)):
                        control_x=start_x-0.5
                    if((start_y < control_y) & (end_y < control_y)):
                        control_y=end_y-0.5
                    elif((start_y > control_y) & (end_y>control_y)):
                        control_y = start_y-0.5
                        
                    
                else:
                    control_x = (start_x + end_x) / 2  + line['line_length']*0.05
                    control_y = (start_y + end_y) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                    if((start_y < control_y) & (end_y < control_y)):
                        control_y=end_y-0.5
                    elif((start_y > control_y) & (end_y>control_y)):
                        control_y = start_y-0.5
                    if((start_x<control_x) & (end_x<control_x)):
                        control_x=end_x-0.5
                    elif((start_x > control_x) & (end_x > control_x)):
                        control_x=start_x-0.5

                curve_x,curve_y = bezier_curve(np.round(start_x).astype(int), np.round(start_y).astype(int), 
                                               np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                               np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
                #plt.plot(curve_x, curve_y, alpha=0.7)
                #plt.xlim(0, box_size)
                #plt.ylim(0, box_size)
                
                df_linTemp = pd.DataFrame({
                    'line_no': i,
                    'line_intensity': line['line_intensity'],
                    'x_positions': curve_x,
                    'y_positions': curve_y
                })
                
                df_data = pd.concat([df_data, df_linTemp], ignore_index=True)
            
            #go as normal on putting a line on another line
            else:
                if(i!=1):
                    randL = np.random.randint(0,i-1) #i-1
                else:
                    randL = i-1
                counter = 0
                #check that the current line to be intersected are not short, as should choose another then
                while((i!=1) & (lines_properties[randL]['line_length']<box_size*0.2)):
                    randL = np.random.randint(0,i-1)
                    counter += 1
                    if counter >= num_lines+10:
                        break
                    
                # check that angles are not too alike
                while(abs(lines_properties[randL]['angle'] - line['angle'])<=np.pi*0.1):
                    #change the angle value
                    line['angle'] = np.random.uniform(0,np.pi)
                    
                start_x, start_y = 0, 0
                end_x = start_x + line['line_length'] * np.cos(line['angle'])
                end_y = start_y + line['line_length'] * np.sin(line['angle'])
                
                curve_x=np.zeros(0)
                while(len(curve_x)<=20):
                    x_or_y = np.random.randint(0,2)
                    if(x_or_y==0):
                        control_x = (start_x + end_x) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                        control_y = (start_y + end_y) / 2 + line['line_length']*0.05
                        if((start_x<control_x) & (end_x<control_x)):
                            control_x=end_x-0.5
                        elif((start_x > control_x) & (end_x > control_x)):
                            control_x=start_x-0.5
                        if((start_y < control_y) & (end_y < control_y)):
                            control_y=end_y-0.5
                        elif((start_y > control_y) & (end_y>control_y)):
                            control_y = start_y-0.5
                    else:
                        control_x = (start_x + end_x) / 2 + line['line_length']*0.05
                        control_y = (start_y + end_y) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                        if((start_x<control_x) & (end_x<control_x)):
                            control_x=end_x-0.5
                        elif((start_x > control_x) & (end_x > control_x)):
                            control_x=start_x-0.5
                        if((start_y < control_y) & (end_y < control_y)):
                            control_y=end_y-0.5
                        elif((start_y > control_y) & (end_y>control_y)):
                            control_y = start_y-0.5
                    
                    curve_x,curve_y = bezier_curve(np.round(start_x).astype(int), np.round(start_y).astype(int), 
                                                   np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                                   np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
                    
                # check to see if the starting point of the new line is actually inside the box
                #ntersect point on previous line
                linenotfound=True
                
                randomPoint = np.random.randint(10,len(df_data[df_data['line_no']==randL]['x_positions'])-10)
                newx,newy = np.asarray(df_data[(df_data['line_no']==randL)][['x_positions','y_positions']])[randomPoint]
                # intersect point on new line
                randomPointF = np.random.randint(10,len(curve_x)-10)
                start_x,start_y = newx - curve_x[randomPointF], newy - curve_y[randomPointF]
                if((0 <= start_x < box_size) & (0 <= start_y < box_size)):
                    linenotfound=False 
                
                # if the start is not inside the point, lets find another intersection point
                counter = 0
                counterLeave = 0
                while(linenotfound):
                    
                    if(counter > 10):
                        if(i!=1):
                            randL = np.random.randint(0,i-1) #i-1
                        else:
                            randL = i-1
                        # check that angles are not too alike
                        while(abs(lines_properties[randL]['angle'] - line['angle'])<=np.pi*0.1):
                            #change the angle value
                            line['angle'] = np.random.uniform(0,np.pi)
                        counter = 0
                        
                    randomPoint = np.random.randint(10,len(df_data[df_data['line_no']==randL]['x_positions'])-10)
                    newx,newy = np.asarray(df_data[(df_data['line_no']==randL)][['x_positions','y_positions']])[randomPoint]
                    # intersect point on new line
                    randomPointF = np.random.randint(10,len(curve_x)-10)
                    start_x,start_y = newx - curve_x[randomPointF], newy - curve_y[randomPointF]
                    
                    if(((0 <= start_x <= box_size) & (0 <= start_y <= box_size))):
                        linenotfound=False
                    
                    counter += 1
                    counterLeave += 1
                    # if this does not work, place a line randomly and go out of this loop
                    if(counterLeave ==100):
                        linenotfound = False
                        start_x, start_y = np.random.randint(0, box_size/2, 2)
                        
                        end_x = start_x + line['line_length'] * np.cos(line['angle'])
                        end_y = start_y + line['line_length'] * np.sin(line['angle'])
                        
                        #check if first line is inside the box
                        while((0>end_x) & (end_x>500) & (0>end_y) & (end_y>500)):
                            start_x, start_y = np.random.randint(0, box_size, 2)
                            
                            end_x = start_x + line['line_length'] * np.cos(line['angle'])
                            end_y = start_y + line['line_length'] * np.sin(line['angle'])
                            
                        x_or_y = np.random.randint(0,2)
                        if(x_or_y==0):
                            control_x = (start_x + end_x) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                            control_y = (start_y + end_y) / 2 + line['line_length']*0.05
                            if((start_x<control_x) & (end_x<control_x)):
                                control_x=end_x-0.5
                            elif((start_x > control_x) & (end_x > control_x)):
                                control_x=start_x-0.5
                            if((start_y < control_y) & (end_y < control_y)):
                                control_y=end_y-0.5
                            elif((start_y > control_y) & (end_y>control_y)):
                                control_y = start_y-0.5
                                
                            
                        else:
                            control_x = (start_x + end_x) / 2  + line['line_length']*0.05
                            control_y = (start_y + end_y) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                            if((start_y < control_y) & (end_y < control_y)):
                                control_y=end_y-0.5
                            elif((start_y > control_y) & (end_y>control_y)):
                                control_y = start_y-0.5
                            if((start_x<control_x) & (end_x<control_x)):
                                control_x=end_x-0.5
                            elif((start_x > control_x) & (end_x > control_x)):
                                control_x=start_x-0.5

                        curve_x,curve_y = bezier_curve(np.round(start_x).astype(int), np.round(start_y).astype(int), 
                                                       np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                                       np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
                        #plt.plot(curve_x, curve_y, alpha=0.7)
                        #plt.xlim(0, box_size)
                        #plt.ylim(0, box_size)
                        
                        df_linTemp = pd.DataFrame({
                            'line_no': i,
                            'line_intensity': line['line_intensity'],
                            'x_positions': curve_x,
                            'y_positions': curve_y
                        })
                        
                        df_data = pd.concat([df_data, df_linTemp], ignore_index=True)
                
                
                
                curve_xF,curve_yF = curve_x + start_x, curve_y + start_y
                
                #plt.plot(curve_xF, curve_yF, alpha=0.7)
                #plt.xlim(0, box_size)
                #plt.ylim(0, box_size)
                
                df_linTemp = pd.DataFrame({
                    'line_no': i,
                    'line_intensity': line['line_intensity'],
                    'x_positions': curve_xF,
                    'y_positions': curve_yF
                })
                
                df_data = pd.concat([df_data, df_linTemp], ignore_index=True)


    return df_data
        




def generate_images_from_data(df_timeseries,noiseBlur,noiseBack,box_size,path,filename,noise='yes'):
    '''

    Parameters
    ----------
    df_timeseries : DataFrame
        DESCRIPTION.
    noiseBlur : int
        The gaussian noise the lines will be smoothed with. This value should typically be around 1-2.
        If checking for extreme fits, set to higher values.
    noiseBack : float
        This is the background noise added. This value should typically be around 0.005.
    box_size : TYPE
        This is the given size of the image, should be set to be the same as from the function used to generate the dataset given as input.

    Returns
    -------
    imgNoise : array
        A numpy array of the df_timeseries. This array has added noise and random blobs in it.

    '''
    #number of frames
    framesL = len(np.unique(df_timeseries['frame']))
    frames = np.unique(df_timeseries['frame'])
    
    img_all = np.zeros(((framesL, box_size, box_size)), dtype=np.double)
    imgNoise = img_all.copy()
    # round x and y poistions to integer
    df_timeseries['x_posR'] = np.round(df_timeseries['x_positions']).astype(int)
    df_timeseries['y_posR'] = np.round(df_timeseries['y_positions']).astype(int)
    # filter away all values outside our frame of view
    df_timeseries = df_timeseries[(df_timeseries['x_posR'] >= 0) & (df_timeseries['x_posR'] < box_size)]
    df_timeseries = df_timeseries[(df_timeseries['y_posR'] >= 0) & (df_timeseries['y_posR'] < box_size)]
    
    df_timeseries.to_csv(path + filename + '_line_position.csv', index=False)  
    
    for i,d in zip(frames,range(framesL)):
        lines = np.unique(df_timeseries[(df_timeseries['frame']==i)]['line_no'])
        if(noise=='yes'):
            maybe_blobs = np.random.randint(int(box_size/10))
            for l in range(maybe_blobs):
                if(np.random.uniform(0,1)>=0.9):
                    #print('blob!',i)
                    posx,posy = np.random.randint(0+4,box_size-4,2)
                    rc = disk((posx, posy), np.random.randint(1,5))
                    img_all[d,rc[0],rc[1]] = np.random.uniform(0.1,0.4)
        lineInt = df_timeseries[df_timeseries['frame']==i].groupby('line_no')['line_intensity'].first()
        for m in lines:
            rxy = df_timeseries[(df_timeseries['frame']==i) & (df_timeseries['line_no']==m)][['x_posR','y_posR']].to_numpy()
            img_all[d,rxy[:,1],rxy[:,0]] = lineInt[m] + img_all[d,rxy[:,1],rxy[:,0]]
            
        # normalise to 1
        img = img_all[d]
        imgf = 1 * (img - img.min()) / (img.max() - img.min()) 
        if(noise=='yes'):
            # now that the 2d array has been created, we can add noise
            # blur and background noise
            imgNoise[d] = skimage.filters.gaussian(imgf,noiseBlur)   
            # must be normalised back to 1
            imgNoise[d] = 1 * (imgNoise[d] - imgNoise[d].min()) / (imgNoise[d].max() - imgNoise[d].min()) 
            
            imgNoise[d] = skimage.util.random_noise(imgNoise[d], mode='gaussian', rng=None, clip=True,mean=0, var=noiseBack).astype(dtype='float32')
            
            #normalise to 0-255, this is needed for conversion to tiff file
            imgNoise[d] = 255 * (imgNoise[d] - imgNoise[d].min()) / (imgNoise[d].max() - imgNoise[d].min()) 
        else:
            imgNoise[d] = imgf*255
            

   

    imgNoise = imgNoise.astype(np.uint8)

    tifffile.imwrite(path+filename + '.tiff', imgNoise, photometric='minisblack',imagej=True)

        
    return imgNoise
     

'''
###############################################################################

Function to identify filamentous structures

###############################################################################
'''

def identification_LAP(csv_list,files_csv,savepathLines,graph_path,graph_list,pos_list,box_size):
    
    for k in range(len(csv_list)):
        results = []
        print(k)
        #load in csv graph and dfpos
        df_Ori = pd.read_csv(files_csv+csv_list[k])
        graphTagg = pd.read_pickle(graph_path+graph_list[k])
        posL = pd.read_pickle(graph_path+pos_list[k])
        
        # filter away all values outside our frame of view
        df_Ori = df_Ori[(df_Ori['x_posR'] >= 0) & (df_Ori['x_posR'] < box_size)]
        df_Ori = df_Ori[(df_Ori['y_posR'] >= 0) & (df_Ori['y_posR'] < box_size)]

        
        frames = np.unique(df_Ori['frame'])
        
        for zs,zx in zip(frames,range(len(frames))):

            df_timeseries = df_Ori[df_Ori['frame']==zs].copy()
            if(graphTagg[zx]!=0):
                graphTagg[zx].edges(data=True)
                postLis = posL[zx]
                
                filaments = nx.get_edge_attributes(graphTagg[zx], 'filament')
                filamentsLength = nx.get_edge_attributes(graphTagg[zx], 'fdist')
                
                # Extract unique values from the dictionary values
                fValues = set(filaments.values())
                trueSLst = df_timeseries['line_no'].unique()
                fLst = list(fValues)
                # costmatrix
                costM = np.zeros((len(trueSLst),len(fValues)))
                for nodeU in list(fValues):
                    #find all node pos for edges marked as certain FS
                    x_entries = list({key: value for key, value in filaments.items() if value == nodeU})
                    
                    saved_values = []
                    saved_values_length = []
                    for z in range(len(x_entries)):
                        
                        node1 = postLis[x_entries[z][0]]-1
                        node2 = postLis[x_entries[z][1]]-1
                        
                        node1x3,node1y3 = np.array([node1[0]-3, node1[0]-2, node1[0]-1, node1[0], node1[0]+1, node1[0]+2, node1[0]+3]), np.array([node1[1]-3, node1[1]-2, node1[1]-1, node1[1], node1[1]+1, node1[1]+2, node1[1]+3])
                        node2x3,node2y3 = np.array([node2[0]-3, node2[0]-2, node2[0]-1, node2[0], node2[0]+1, node2[0]+2, node2[0]+3]), np.array([node2[1]-3, node2[1]-2, node2[1]-1, node2[1], node2[1]+1, node2[1]+2, node2[1]+3])
                        
                        node_valX = []
                        xy_coordsX = []
                        node_valY = []
                        xy_coordsY = []
                        for i in range(len(node1x3)):
                            for m in range(len(node1y3)):
                                # node 1
                                xdf = df_timeseries[(df_timeseries['x_positions'] == node1x3[i]) & (df_timeseries['y_positions'] == node1y3[m])]
                                #print(i,m,xdf)
                                if(not xdf.empty):
                                    node_valX.append(xdf['line_no'].values[0])
                                    xy_coordsX.append(xdf[['x_posR','y_posR']].values)
                                    
                                # node 2
                                ydf = df_timeseries[(df_timeseries['x_positions'] == node2x3[i]) & (df_timeseries['y_positions'] == node2y3[m])]
                                #print(i,m,ydf)
                                if(not ydf.empty):
                                    node_valY.append(ydf['line_no'].values[0])
                                    xy_coordsY.append(ydf[['x_posR','y_posR']].values)
                      
                        # Find common elements     
                        common_elements = list(set(node_valX).intersection(node_valY))
                        
                        
                        saved_values.append(common_elements)
                        saved_values_length.append(filamentsLength[x_entries[z]])
                        
                    # if no matching node pairs exist that overlap FS marks in timeseries
                    is_empty = all(len(sublist) == 0 for sublist in saved_values)
                    if(not is_empty):
                        lengthEdges = np.asarray(saved_values_length)
                        value_to_check = np.unique(list(itertools.chain.from_iterable(saved_values)))
                        I_len_FilCoverage = np.zeros(len(value_to_check))
                        I_trueLine = np.zeros(len(value_to_check))
                        i_filLine = np.zeros(len(value_to_check))
                        for ks,l in zip(value_to_check,range(len(value_to_check))):
                            lineLen = len(df_timeseries[(df_timeseries['frame'] == zs) & (df_timeseries['line_no'] == ks)])
                            perEdge = lengthEdges/lineLen
                            # Calculate the total number of sublists
                            # Get indices of sublists that contain the specified value
                            indices = [index for index, sublist in enumerate(saved_values) if ks in sublist]
                            
                            I_len_FilCoverage[l] = np.sum(perEdge[indices])
                            i_filLine[l] = np.sum(lengthEdges[indices])
                            I_trueLine[l] = np.sum(lineLen)
                            trueI = np.where(trueSLst==ks)[0][0]
                            
                            costM[trueI,nodeU] = np.sum(perEdge[indices])
                            
                row_ind, col_ind = linear_sum_assignment(costM,maximize=True)    
                
                definedM = np.zeros(len(row_ind))
                deleteDef = []
                for n in range(len(row_ind)):
                    list_true_line_len = len(df_timeseries[(df_timeseries['frame'] == zs) & (df_timeseries['line_no'] == trueSLst[n])])
                    #find correct matched filament
                    x_entries = list({key: value for key, value in filaments.items() if value == fLst[col_ind[n]]})
                    list_filament_line_len = sum(filamentsLength[key] for key in x_entries if key in filamentsLength)
                    list_FilCoverage = costM[row_ind[n],col_ind[n]]
                    definedM[n] = fLst[col_ind[n]]
                    if(costM[row_ind[n],col_ind[n]]==0):
                        deleteDef = np.append(deleteDef,int(n))
                        bestM = None
                        list_filament_line_len = 0
                    else:
                        bestM = fLst[col_ind[n]]

                        
                    results.append({'frame': zs,
                                    'true index': trueSLst[n],
                                    'match index': bestM,
                                    'FS_coverage': list_FilCoverage,
                                    'FS_true_len': list_true_line_len,
                                    'FS_found_len': list_filament_line_len,
                                    'overlap ratio': list_FilCoverage
                                    })
                    
                if(len(deleteDef)!=0):
                    definedM = np.delete(definedM,deleteDef.astype(int))
                main_list = list(set(fLst) - set(definedM))
                for x in range(len(main_list)):
                    list_true_line_len = None
                    #find correct matched filament
                    x_entries = list({key: value for key, value in filaments.items() if value == main_list[x]})
                    list_filament_line_len = sum(filamentsLength[key] for key in x_entries if key in filamentsLength)
                    
                    # find out if this defined filament oerlapped a true filament
                    indices = np.where(np.asarray(fLst) == main_list[x])
                    overlap = np.max(costM[:,indices])
                    
                    results.append({'frame': zs,
                                    'true index': None,
                                    'match index': main_list[x],
                                    'FS_coverage': 0,
                                    'FS_true_len': list_true_line_len,
                                    'FS_found_len': list_filament_line_len,
                                    'overlap ratio': overlap
                                    })
                    
                  
        results_df = pd.DataFrame(results)
        
        results_df.to_csv(savepathLines + '{0}_df_line_comparison.csv'.format(k), index=False)  
    return 


def identification(csv_list,files_csv,savepathLines,graph_path,graph_list,pos_list,box_size):
    for k in range(len(csv_list)):
        #load in csv graph and dfpos
        df_Ori = pd.read_csv(files_csv+csv_list[k])
        graphTagg = pd.read_pickle(graph_path+graph_list[k])
        posL = pd.read_pickle(graph_path+pos_list[k])
        
        # filter away all values outside our frame of view
        df_Ori = df_Ori[(df_Ori['x_posR'] >= 0) & (df_Ori['x_posR'] < box_size)]
        df_Ori = df_Ori[(df_Ori['y_posR'] >= 0) & (df_Ori['y_posR'] < box_size)]
        
        dfLineProp = pd.DataFrame()
        
        frames = np.unique(df_Ori['frame'])
        for zs,zx in zip(frames,range(len(frames))):
        
            df_timeseries = df_Ori[df_Ori['frame']==zs].copy()
            if(graphTagg[zx]!=0):
                graphTagg[zx].edges(data=True)
                postLis = posL[zx]
                
                filaments = nx.get_edge_attributes(graphTagg[zx], 'filament')
                filamentsLength = nx.get_edge_attributes(graphTagg[zx], 'fdist')
                
                # Extract unique values from the dictionary values
                fValues = set(filaments.values())
                list_correct_wrong_mark =  np.ones(len(fValues))*2
                list_FilCoverage = np.zeros(len(fValues))
                list_true_line_len = np.zeros(len(fValues))
                list_filament_line_len = np.zeros(len(fValues))
                for nodeU in list(fValues):
                    #find all node pos for edges marked as certain FS
                    x_entries = list({key: value for key, value in filaments.items() if value == nodeU})
                    
                    saved_values = []
                    saved_values_length = []
                    for z in range(len(x_entries)):
                        
                        node1 = postLis[x_entries[z][0]]-1
                        node2 = postLis[x_entries[z][1]]-1
                        
                        node1x3,node1y3 = np.array([node1[0]-2,node1[0]-1,node1[0],node1[0]+1, node1[0]+2]), np.array([node1[1]-2, node1[1]-1,node1[1],node1[1]+1,node1[1]+2])
                        node2x3,node2y3 = np.array([node2[0]-2, node2[0]-1,node2[0],node2[0]+1, node2[0]+2]), np.array([node2[1]-2, node2[1]-1,node2[1],node2[1]+1, node2[1]+2])
                        
                        node_valX = []
                        xy_coordsX = []
                        node_valY = []
                        xy_coordsY = []
                        for i in range(len(node1x3)):
                            for m in range(len(node1y3)):
                                # node 1
                                xdf = df_timeseries[(df_timeseries['x_positions'] == node1x3[i]) & (df_timeseries['y_positions'] == node1y3[m])]
                                #print(i,m,xdf)
                                if(not xdf.empty):
                                    node_valX.append(xdf['line_no'].values[0])
                                    xy_coordsX.append(xdf[['x_posR','y_posR']].values)
                                    
                                # node 2
                                ydf = df_timeseries[(df_timeseries['x_positions'] == node2x3[i]) & (df_timeseries['y_positions'] == node2y3[m])]
                                #print(i,m,ydf)
                                if(not ydf.empty):
                                    node_valY.append(ydf['line_no'].values[0])
                                    xy_coordsY.append(ydf[['x_posR','y_posR']].values)
                      
                        # Find common elements     
                        common_elements = list(set(node_valX).intersection(node_valY))
                        
                        
                        saved_values.append(common_elements)
                        saved_values_length.append(filamentsLength[x_entries[z]])
                        
                    # if no matching node pairs exist that overlap FS marks in timeseries
                    is_empty = all(len(sublist) == 0 for sublist in saved_values)
                    if(is_empty):
                        list_correct_wrong_mark[nodeU] = 0
                        list_FilCoverage[nodeU] = 0
                        list_true_line_len[nodeU] = 0
                        list_filament_line_len[nodeU] = np.sum(saved_values_length)
                    else:
                        lengthEdges = np.asarray(saved_values_length)
                        value_to_check = np.unique(list(itertools.chain.from_iterable(saved_values)))
                        I_len_FilCoverage = np.zeros(len(value_to_check))
                        I_trueLine = np.zeros(len(value_to_check))
                        i_filLine = np.zeros(len(value_to_check))
                        for ks,l in zip(value_to_check,range(len(value_to_check))):
                            lineLen = len(df_timeseries[(df_timeseries['frame'] == zs) & (df_timeseries['line_no'] == ks)])
                            perEdge = lengthEdges/lineLen
                            # Calculate the total number of sublists
                            # Get indices of sublists that contain the specified value
                            indices = [index for index, sublist in enumerate(saved_values) if ks in sublist]
                            
                            I_len_FilCoverage[l] = np.sum(perEdge[indices])
                            i_filLine[l] = np.sum(lengthEdges[indices])
                            I_trueLine[l] = np.sum(lineLen)
                            
                        TrueIndex = np.argmax(I_len_FilCoverage)
                        list_FilCoverage[nodeU] = I_len_FilCoverage[TrueIndex]
                        list_true_line_len[nodeU] = I_trueLine[TrueIndex]
                        list_filament_line_len[nodeU] = i_filLine[TrueIndex]
                        
                        #remove the line that has been matched with from the dataframe of true lines
                        df_timeseries = df_timeseries[df_timeseries['line_no'] != value_to_check[TrueIndex]]
            
            df_linTemp = pd.DataFrame({
                'frame': zs,
                'FS_coverage': list_FilCoverage,
                'FS_true_len': list_true_line_len,
                'FS_found_len': list_filament_line_len
            })
            
            dfLineProp = pd.concat([dfLineProp, df_linTemp], ignore_index=True)
        
        dfLineProp.to_csv(savepathLines + '{0}_df_line_comparison.csv'.format(k), index=False)  
    return 


'''
###############################################################################

calculate difference from original image

###############################################################################
'''

def signaltonoise(true_img,img):
    col,row = np.where((true_img>0))
    list_im =[]
    for i in range(len(col)):
        list_im.append(img[col[i],row[i]])
    m = np.mean(list_im)
    sd = img[400:499,0:99].std()
    if(sd==0):
        snr = 0
    else:
        snr = m/sd
    return snr


def calcDiff(img_o,mask,df_pos):
    
    # pad images and df_pos
    img_o = np.pad(img_o, 3 ,mode = 'constant')
    mask = np.pad(mask, 3 ,mode = 'constant')
    # calculation of image created from masks belonging to edges
    imgMF = np.zeros((img_o.shape))
    for i in df_pos['map value']:
        imgMF += mask == i
    countMask = np.sum(imgMF)
    '''
    plt.figure()
    plt.imshow(imgMF)
    '''
    # the original skeleton version of image / true location
    imgOri = (img_o>0)*1
    
    count = 0
    #perfect overlap search
    (rows,cols) = np.nonzero((imgMF>0)*1)
    for k in range(len(rows)):
        r = rows[k]
        c = cols[k]
        if(imgOri[r,c]==1):
            imgMF[r,c] = 0
            imgOri[r,c] = 0
            count += 1
    
     # 1 radii search
    (rows,cols) = np.nonzero((imgMF>0)*1)
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
            imgMF[r,c] = 0
            count += 1
    
    # 2 radii search
    (rows,cols) = np.nonzero((imgMF>0)*1)
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
            imgMF[r,c] = 0
            count += 1
            
    # 3 radii search
    (rows,cols) = np.nonzero((imgMF>0)*1)
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
            imgMF[r,c] = 0
            count += 1
            
    FP = np.sum(imgMF)
    correctC = np.sum((img_o>0)*1)        
    perC = count/correctC*100
    FN = (correctC - count)/correctC*100
    FP_trueimg = FP/correctC*100
    
    return correctC,countMask, count, perC, FP_trueimg, FN

def JI(TP,FP,FN):
    JI = TP/(TP + FP + FN)
    return JI

def draw_graph_filament_nocolor(image,graph,pos,title,value):
    edges,values = zip(*nx.get_edge_attributes(graph,value).items())
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(image,cmap='gray_r')
    vmin=min(values)
    vmax=max(values)+1
    if(vmax%2==0):
        vmax +=1
    cmap=plt.get_cmap('tab10',vmax)
    nx.draw(graph,pos, edge_cmap=cmap, edge_color=values,node_size=35,width=10,alpha=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    #plt.colorbar(sm,orientation='horizontal')
    plt.tight_layout()
    return

'''
###############################################################################

Function to use GraFT

###############################################################################
'''

def create_tiffimgs(imgs_list,path_imgs):
    countImgsplot = 0
    for zx in range(len(imgs_list)):
        print(zx)
            
        with tifffile.TiffFile(path_imgs+imgs_list[zx]) as tif:
            img_o = tif.asarray()
            
        for i in range(len(img_o)):
            plt.figure(figsize=(10,10))
            plt.imshow(img_o[i],cmap='gray_r')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(path_imgs+'imgs/img{0}.png'.format(countImgsplot))
            countImgsplot += 1
        
            plt.close('all')
    return

def use_GraFT(imgs_list,path_imgs,box_size,graphSave,size,eps,thresh_top,sigma,small):

    countGraph = 0
    JI_all = []
    for zx in range(len(imgs_list)):
        print(zx)
            
        with tifffile.TiffFile(path_imgs+imgs_list[zx]) as tif:
            img_o = tif.asarray()
            
        filenameOri = re.sub(r'.tiff', '', imgs_list[zx])
        if 'nonoise' in filenameOri:
            img_lines = img_o.copy()
        else:
            with tifffile.TiffFile(path_imgs+filenameOri+'_nonoise.tiff') as tif:
                img_lines = tif.asarray()
                
        img_linesPad = np.pad(img_lines, ((0, 0), (1, 1), (1, 1)), mode = 'constant')
        
        graph_s = [0]*len(img_o)
        posL = [0]*len(img_o)
        imgSkel = [0]*len(img_o)
        imgAF = [0]*len(img_o)
        imgBl = [0]*len(img_o)
        imF = [0]*len(img_o)
        mask = [0]*len(img_o)
        df_pos = [0]*len(img_o)
        graphD = [0]*len(img_o)
        lgG = [0]*len(img_o)
        lgG_V = [0]*len(img_o)
        graphTagg = [0]*len(img_o)
        
        count = [0]*len(img_o)
        countIm = [0]*len(img_o)
        correctCount =[0]*len(img_o)
        TPI = np.zeros(len(img_o))
        FNI = np.zeros(len(img_o))
        FPI = np.zeros(len(img_o))
        
        for q in range(len(img_o)):
            print(q)
            # 0) create graph
            graph_s[q], posL[q], imgSkel[q], imgAF[q], imgBl[q],imF[q],mask[q],df_pos[q] = utilsF.creategraph(img_o[q],size=size,eps=eps,thresh_top=thresh_top,sigma=sigma,small=small)
            if(graph_s[q]!=0):
                
                utilsF.draw_graph(imgSkel[q],graph_s[q],posL[q],"untagged graph")
                #plt.savefig(pathsave+'noise/n_graphs/untaggedgraph{0}.png'.format(q))
                # 1) find all dangling edges and mark them
                graphD[q] = utilsF.dangling_edges(graph_s[q].copy())
                # 2) create line graph
                lgG[q] = nx.line_graph(graph_s[q].copy())
                # 3) calculate the angles between two edges from the graph that is now represented by edges in the line graph
                lgG_V[q] = utilsF.lG_edgeVal(lgG[q].copy(),graphD[q],posL[q])
                # 4) run depth first search
                graphTagg[q] = utilsF.dfs_constrained(graph_s[q].copy(),lgG_V[q].copy(),imgBl[q],posL[q],140,3,angle_free='yes') 
                
                draw_graph_filament_nocolor(imgSkel[q],graphTagg[q],posL[q],"",'filament')
                plt.savefig(path_imgs+'graph/n_graphs/graph{0}.png'.format(countGraph))
       
                #print('filament defined: ',len(np.unique(np.asarray(list(graphTagg[q].edges(data='filament')))[:,2])))
                
                correctCount[q],countIm[q],count[q],TPI[q],FPI[q],FNI[q] = calcDiff(img_linesPad[q],mask[q],df_pos[q])
            
            else:
                #switchEmpty=1
                print('graph is empty')
            plt.close('all')
            countGraph += 1
        '''
        print('overlap count:',np.mean(count), np.std(count))
        print('TP ',np.mean(TPI),np.std(TPI))
        print('FP ',np.mean(FPI), np.std(FPI))
        print('FN ',np.mean(FNI),np.std(FNI))
        '''
        '''
        if(switchEmpty==1):
            posL = [element for element in posL if not np.array_equal(element, 0)]
            graphTagg = [element for element in graphTagg if not np.array_equal(element, 0)]
        '''

            
        pickle.dump(posL, open(graphSave+'{0}_posL.gpickle'.format(zx), 'wb'))
        pickle.dump(graphTagg, open(graphSave+'{0}_graphTagg.gpickle'.format(zx), 'wb'))
        
        
        JI_list = JI(TPI,FPI,FNI)
        JI_all.append(JI_list)
        df_JI = pd.DataFrame({
            'frame': np.arange(len(img_o)),
            'JI': JI_list
        })
        df_JI.to_csv(path_imgs + 'JI/{0}_JI.csv'.format(zx), index=False) 

    return

'''
###############################################################################

Function to calculate density

###############################################################################
'''

def graph_density(graph_list,graph_path):
    

    densG = []
    degreeG = []
    for k in range(len(graph_list)):
        
        graphTagg = pd.read_pickle(graph_path+graph_list[k])
        
        densg=np.zeros(len(graphTagg))
        degreeAv=np.zeros(len(graphTagg))
        for zx in range(len(graphTagg)):
            
            densg[zx] = nx.density(graphTagg[zx])
            
            degreeAv[zx] = 2*graphTagg[zx].number_of_edges() / float(graphTagg[zx].number_of_nodes())
            
        densG = np.append(densG,densg)
        degreeG = np.append(degreeG,degreeAv)
    df_densAllGraph = pd.DataFrame({
        'frame': np.arange(0,len(densG)),
        'Density graph': densG,
        'Average degree': degreeG
    })
    return df_densAllGraph

def density(filescsv,csvlist,box_size,len_files,imgSavePath):
    densAll=[]
    densAllImg=[]
    
    for kl in range(len(csvlist)):
        
        
        df_Ori = pd.read_csv(filescsv+csvlist[kl])
        
        # filter away all values outside our frame of view
        df_Ori = df_Ori[(df_Ori['x_positions'] >= 0) & (df_Ori['x_positions'] < box_size)]
        df_Ori = df_Ori[(df_Ori['y_positions'] >= 0) & (df_Ori['y_positions'] < box_size)]
        
        frames = np.unique(df_Ori['frame'])
        density=np.zeros(len(frames))
        densityImg=np.zeros(len(frames))
        print(kl,len(frames))
        for zs,zx in zip(frames,range(len(frames))):
        
            df_Int = df_Ori[df_Ori['frame']==zs].copy()
            
            
            img = np.zeros(((box_size, box_size)),dtype=np.uint8)
            img[df_Int['x_positions'],df_Int['y_positions']] = 1
            
            img1 = img*255
            tifffile.imwrite(imgSavePath+'{0}_img_nonoise'.format(zs) + '.tiff', img1, photometric='minisblack',imagej=True)
            
            #maskConHull = convex_hull_image(img)
            
            #density[zx] = np.sum(img)/np.sum(maskConHull)
            densityImg[zx] = np.sum(img)/box_size**2
            
        densAll = np.append(densAll,density)
        densAllImg = np.append(densAllImg,densityImg)
        
        
    df_densAll = pd.DataFrame({
        'frame': np.arange(0,len_files),
        #'Density': densAll,
        'Density full image': densAllImg
    })
    
    return df_densAll

