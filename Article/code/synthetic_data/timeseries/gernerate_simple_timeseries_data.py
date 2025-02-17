#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:26:09 2024

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


            
'''
###############################################################################

create simple frames with non overlapping FS in timeseries

###############################################################################
'''

# Helper function to calculate control points
def calculate_control_points(start_x, start_y, end_x, end_y, line_length, curvature_weight, x_or_y):
    """Calculates Bezier curve control points based on x_or_y flag."""
    
    if x_or_y == 0:
        control_x = (start_x + end_x) / 2 + line_length * 0.05 + np.random.uniform(0, curvature_weight)
        control_y = (start_y + end_y) / 2 + line_length * 0.05
    else:
        control_x = (start_x + end_x) / 2 + line_length * 0.05
        control_y = (start_y + end_y) / 2 + line_length * 0.05 + np.random.uniform(0, curvature_weight)

    # Adjust control points to avoid overlap
    if (start_x < control_x) and (end_x < control_x):
        control_x = end_x - 0.5
    elif (start_x > control_x) and (end_x > control_x):
        control_x = start_x - 0.5

    if (start_y < control_y) and (end_y < control_y):
        control_y = end_y - 0.5
    elif (start_y > control_y) and (end_y > control_y):
        control_y = start_y - 0.5

    return control_x, control_y

# Helper function to generate Bezier curve points
def generate_bezier_curve(start_x, start_y, control_x, control_y, end_x, end_y):
    """Generates points for a Bezier curve."""
    return bezier_curve(
        np.round(start_x).astype(int), np.round(start_y).astype(int),
        np.round(control_x).astype(int), np.round(control_y).astype(int),
        np.round(end_x).astype(int), np.round(end_y).astype(int), 4
    )


def generate_filamentous_structure(box_size, num_lines,frame_no, movement, curvature_factor):
    
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

    # Initial setup for plots and data storage
    #plt.figure(figsize=(10, 10))
    df_data = pd.DataFrame()
    df_linedata = pd.DataFrame()
    max_possible_length = box_size

    # Generate initial lines and their properties
    lines_properties = []
    line_lengthL = np.zeros(num_lines)
    for i in range(num_lines):
        
        # make relative shorter filaments more likely
        # perhaps use power law
        line_length = 0
        while((line_length<10) or (line_length>=max_possible_length)):        
            line_length = np.random.exponential(scale=max_possible_length * 0.5)
            
            #make first line longer than box_size*0.2
        if((i == 0) and (line_length > box_size * 0.2)):
            line_length = box_size*0.4
            
        normalized_influence = (line_length - 25) / (box_size - 25)
        line_intensity = 0.5 + (normalized_influence * 0.5) * np.random.random()
        
        #line_intensity = min((line_length * (1 + np.random.uniform(0., 0.8))), max_possible_length) / max_possible_length
        # random weight to be added to beizer control points. This value should max go to line_intensity
        random_weight = np.max([line_intensity - np.random.uniform(0, line_intensity), 0.2])
        curvature_weight = curvature_factor * (1 - random_weight)
        
        # Initial angle biased towards the major axis
        # if x, normal around 0, with scale pi/4, else normal around pi/2 (90) 
        angle = np.random.uniform(0,np.pi) # 0-180
        x_or_y = np.random.randint(0,2)
        
        start_x, start_y = np.random.randint(50, box_size-51, 2)
        
        movement_direction = np.random.randint(0,2)

        # for the first frame, create bezier curves directly, so we know where to position next line
        if(i==0):
            end_x = start_x + line_length * np.cos(angle)
            end_y = start_y + line_length * np.sin(angle)
            
            #check if first line is inside the box
            
            while((0 > end_x) or (end_x > box_size) or (0 > end_y) or (end_y > box_size)):
                start_x, start_y = np.random.randint(50, box_size-51, 2)
                
                end_x = start_x + line_length * np.cos(angle)
                end_y = start_y + line_length * np.sin(angle)
            
            
            control_x, control_y = calculate_control_points(start_x, start_y, end_x, end_y, line_length, curvature_weight, x_or_y)
            
            curve_x,curve_y = generate_bezier_curve(start_x, start_y, control_x, control_y, end_x, end_y)
            
            lines_properties.append({
                'line_length': line_length,
                'line_intensity': line_intensity,
                'angle': angle,
                'curvature': curvature_weight,
                'x-y': x_or_y,
                'x_start': start_x,
                'y_start': start_y,
                'direction': movement_direction,
                'frame': 0
            })
            line_lengthL[i] = line_length 
            
            df_linTemp = pd.DataFrame({
                'frame': 0,
                'line_no': i,
                'line_intensity': line_intensity,
                'x_positions': curve_x,
                'y_positions': curve_y
            })
            
            df_data = pd.concat([df_data, df_linTemp], ignore_index=True)
            
        else:
            #create probability of a filament lying a random space
            random_number = np.random.uniform()
    
            if(random_number>=0.6):
                
                end_x = start_x + line_length * np.cos(angle)
                end_y = start_y + line_length * np.sin(angle)
                
                #check if first line is inside the box
                while((0 > end_x) or (end_x > 500) or (0 > end_y) or (end_y > 500)):
                #while((0>end_x) & (end_x>500) & (0>end_y) & (end_y>500)):
                    start_x, start_y = np.random.randint(50, box_size-51, 2)
                    
                    end_x = start_x + line_length * np.cos(angle)
                    end_y = start_y + line_length * np.sin(angle)
                    
                x_or_y = np.random.randint(0,2)
                
                control_x, control_y = calculate_control_points(start_x, start_y, end_x, end_y, line_length, curvature_weight, x_or_y)
    
                curve_x,curve_y = bezier_curve(np.round(start_x).astype(int), np.round(start_y).astype(int), 
                                               np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                               np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
                
                lines_properties.append({
                    'line_length': line_length,
                    'line_intensity': line_intensity,
                    'angle': angle,
                    'curvature': curvature_weight,
                    'x-y': x_or_y,
                    'x_start': start_x,
                    'y_start': start_y,
                    'direction': movement_direction,
                    'frame': 0
                })
                line_lengthL[i] = line_length 
                
                df_linTemp = pd.DataFrame({
                    'frame': 0,
                    'line_no': i,
                    'line_intensity': line_intensity,
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
                    if counter >= num_lines+20:
                        randL = 1
                    
                # check that angles are not too alike
                while(abs(lines_properties[randL]['angle'] - angle)<=np.pi*0.1):
                    #change the angle value
                    angle = np.random.uniform(0,np.pi)
                    
                start_x, start_y = 0, 0
                end_x = start_x + line_length * np.cos(angle)
                end_y = start_y + line_length * np.sin(angle)
                
                curve_x=np.zeros(0)
                while(len(curve_x)<=40):
                    x_or_y = np.random.randint(0,2)
                    
                    control_x, control_y = calculate_control_points(start_x, start_y, end_x, end_y, line_length, curvature_weight, x_or_y)

                    curve_x,curve_y = bezier_curve(np.round(start_x).astype(int), np.round(start_y).astype(int), 
                                                   np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                                   np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
                    
                # check to see if the starting point of the new line is actually inside the box
                #ntersect point on previous line
                linenotfound=True
                
                randomPoint = np.random.randint(10,len(df_data[df_data['line_no']==randL]['x_positions'])-10)
                newx,newy = np.asarray(df_data[(df_data['line_no']==randL)][['x_positions','y_positions']])[randomPoint]
                # intersect point on new line
                randomPointF = np.random.randint(20,len(curve_x)-20)
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
                        while(abs(lines_properties[randL]['angle'] - angle)<=np.pi*0.1):
                            #change the angle value
                            angle = np.random.uniform(0,np.pi)
                        counter = 0
                        
                    randomPoint = np.random.randint(10,len(df_data[df_data['line_no']==randL]['x_positions'])-10)
                    newx,newy = np.asarray(df_data[(df_data['line_no']==randL)][['x_positions','y_positions']])[randomPoint]
                    # intersect point on new line
                    randomPointF = np.random.randint(20,len(curve_x)-20)
                    start_x,start_y = newx - curve_x[randomPointF], newy - curve_y[randomPointF]
                    
                    if(((0 <= start_x <= box_size) & (0 <= start_y <= box_size))):
                        linenotfound=False
                    
                    counter += 1
                    counterLeave += 1
                    # if this does not work, place a line randomly and go out of this loop
                    if(counterLeave ==100):
                        linenotfound = False
                        start_x, start_y = np.random.randint(50, box_size-51, 2)
                        
                        end_x = start_x + line_length * np.cos(angle)
                        end_y = start_y + line_length * np.sin(angle)
                        
                        #check if first line is inside the box
                        while((0 > end_x) or (end_x > 500) or (0 > end_y) or (end_y > 500)):
                            start_x, start_y = np.random.randint(50, box_size-51, 2)
                            
                            end_x = start_x + line_length * np.cos(angle)
                            end_y = start_y + line_length * np.sin(angle)
                            
                        x_or_y = np.random.randint(0,2)
                        
                        control_x, control_y = calculate_control_points(start_x, start_y, end_x, end_y, line_length, curvature_weight, x_or_y)
              
                        curve_x,curve_y = bezier_curve(np.round(start_x).astype(int), np.round(start_y).astype(int), 
                                                       np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                                       np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
                        
                lines_properties.append({
                    'line_length': line_length,
                    'line_intensity': line_intensity,
                    'angle': angle,
                    'curvature': curvature_weight,
                    'x-y': x_or_y,
                    'x_start': start_x,
                    'y_start': start_y,
                    'direction': movement_direction,
                    'frame': 0
                })
                line_lengthL[i] = line_length 
                
                df_linTemp = pd.DataFrame({
                    'frame': 0,
                    'line_no': i,
                    'line_intensity': line_intensity,
                    'x_positions': curve_x,
                    'y_positions': curve_y
                })
                
                df_data = pd.concat([df_data, df_linTemp], ignore_index=True)
                
        
    iL = np.argmax(line_lengthL)
    item = lines_properties[iL]
    lines_properties.pop(iL)
    lines_properties.insert(0,item)
    
    df_linetime = pd.DataFrame(lines_properties)
    
    # add in movement to start and end points for each frame added
    for m in range(1,frame_no):
        
        df_linetimeA = df_linetime[df_linetime['frame']==m-1].copy()
        for kl in df_linetimeA.index.tolist():
            if(df_linetimeA.loc[kl, 'direction'] == 0):
                #move negative, but certain probability of moving positive
                interval=np.asarray([-movement,1])
                if(np.random.uniform(0,1)>0.9):
                    interval=np.asarray([0,movement+1])
            else:
                #positive movement, with certain prob to move negative
                interval=np.asarray([0,movement+1])
                if(np.random.uniform(0,1)>0.9):
                    interval=np.asarray([-movement,1])
                        
            df_linetimeA.loc[kl, ['x_start', 'y_start']] += np.random.randint(low=interval[0], high=interval[1], size=2)
            df_linetimeA['frame']=m
        df_linetime = pd.concat([df_linetime, df_linetimeA], ignore_index=True)
            
        
    for d in range(frame_no):
        
        lines_propertiesA = df_linetime[df_linetime['frame']==d].to_dict(orient='records')
    
        # Place in a bounding box
        for i, line in enumerate(lines_propertiesA):
            # generate the beizer curve
            
            end_x = line['x_start'] + line['line_length'] * np.cos(line['angle'])
            end_y = line['y_start'] + line['line_length'] * np.sin(line['angle'])
            
            if(line['x-y']==0):
                control_x = (line['x_start'] + end_x) / 2 + line['line_length']*0.05 + line['curvature']
                control_y = (line['y_start'] + end_y) / 2 + line['line_length']*0.05
                if((line['x_start']<control_x) & (end_x<control_x)):
                    control_x=end_x-0.5
                elif((line['x_start'] > control_x) & (end_x > control_x)):
                    control_x=line['x_start']-0.5
                if((line['y_start'] < control_y) & (end_y < control_y)):
                    control_y=end_y-0.5
                elif((line['y_start'] > control_y) & (end_y>control_y)):
                    control_y = line['y_start']-0.5
                    
                
            else:
                control_x = (line['x_start'] + end_x) / 2  + line['line_length']*0.05
                control_y = (line['y_start'] + end_y) / 2 + line['line_length']*0.05 + line['curvature']
                if((line['y_start'] < control_y) & (end_y < control_y)):
                    control_y=end_y-0.5
                elif((line['y_start'] > control_y) & (end_y>control_y)):
                    control_y = line['y_start']-0.5
                if((line['x_start']<control_x) & (end_x<control_x)):
                    control_x=end_x-0.5
                elif((line['x_start'] > control_x) & (end_x > control_x)):
                    control_x=line['x_start']-0.5
    
            curve_x,curve_y = bezier_curve(np.round(line['x_start']).astype(int), np.round(line['y_start']).astype(int), 
                                           np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                           np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
            #plt.plot(curve_x, curve_y, alpha=0.7)
            #plt.xlim(0, box_size)
            #plt.ylim(0, box_size)
            
            df_linTemp = pd.DataFrame({
                'frame': d,
                'line_no': i,
                'line_intensity': line['line_intensity'],
                'x_positions': curve_x,
                'y_positions': curve_y
            })
            
            df_linedata = pd.concat([df_linedata, df_linTemp], ignore_index=True)

    return df_linedata



def generate_simple_filamentous_structure(box_size, num_lines,frame_no, movement, curvature_factor):
    
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
        line_length = 0
        while((line_length<10) or (line_length>=100)):        
            line_length = np.random.exponential(scale=max_possible_length * 0.1)

        
        normalized_influence = (line_length - 25) / (box_size - 25)
        line_intensity = 0.5 + (normalized_influence * 0.5) * np.random.random()
        
        #line_intensity = min((line_length * (1 + np.random.uniform(0., 0.8))), max_possible_length) / max_possible_length
        # random weight to be added to beizer control points. This value should max go to line_intensity
        random_weight = np.max([line_intensity - np.random.uniform(0, line_intensity), 0.2])
        curvature_weight = curvature_factor * (1 - random_weight)
        
        # Initial angle biased towards the major axis
        # if x, normal around 0, with scale pi/4, else normal around pi/2 (90) 
        angle = np.random.uniform(0,np.pi) # 0-180
        x_or_y = np.random.randint(0,2)
        
        start_x, start_y = np.random.randint(30, box_size-30, 2)
        
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
            'curvature': curvature_weight,
            'x-y': x_or_y,
            'x_start': start_x,
            'y_start': start_y,
            'frame': 0
        })
        line_lengthL[i] = line_length 
    iL = np.argmax(line_lengthL)
    item = lines_properties[iL]
    lines_properties.pop(iL)
    lines_properties.insert(0,item)
    
    df_linetime = pd.DataFrame(lines_properties)
    
    # add in movement to start and end points for each frame added
    for m in range(frame_no):
        
        df_linetimeA = df_linetime[df_linetime['frame']==m].copy()

        df_linetimeA[['x_start', 'y_start']] = df_linetimeA[['x_start', 'y_start']].apply(lambda x: x + np.random.randint(-movement,movement), axis=1)
        df_linetimeA['frame']=m+1
        df_linetime = pd.concat([df_linetime, df_linetimeA], ignore_index=True)
        
        
    for d in np.unique(df_linetime['frame']):
        
        lines_propertiesA = df_linetime[df_linetime['frame']==d].to_dict(orient='records')
    
        # Place in a bounding box
        for i, line in enumerate(lines_propertiesA):
            # generate the beizer curve
            
            end_x = line['x_start'] + line['line_length'] * np.cos(line['angle'])
            end_y = line['y_start'] + line['line_length'] * np.sin(line['angle'])
            
            
            x_or_y = np.random.randint(0,2)
            if(line['x-y']==0):
                control_x = (line['x_start'] + end_x) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                control_y = (line['y_start'] + end_y) / 2 + line['line_length']*0.05
                if((line['x_start']<control_x) & (end_x<control_x)):
                    control_x=end_x-0.5
                elif((line['x_start'] > control_x) & (end_x > control_x)):
                    control_x=line['x_start']-0.5
                if((line['y_start'] < control_y) & (end_y < control_y)):
                    control_y=end_y-0.5
                elif((line['y_start'] > control_y) & (end_y>control_y)):
                    control_y = line['y_start']-0.5
                    
                
            else:
                control_x = (line['x_start'] + end_x) / 2  + line['line_length']*0.05
                control_y = (line['y_start'] + end_y) / 2 + line['line_length']*0.05 + np.random.uniform(0, line['curvature'])
                if((line['y_start'] < control_y) & (end_y < control_y)):
                    control_y=end_y-0.5
                elif((line['y_start'] > control_y) & (end_y>control_y)):
                    control_y = line['y_start']-0.5
                if((line['x_start']<control_x) & (end_x<control_x)):
                    control_x=end_x-0.5
                elif((line['x_start'] > control_x) & (end_x > control_x)):
                    control_x=line['x_start']-0.5
    
            curve_x,curve_y = bezier_curve(np.round(line['x_start']).astype(int), np.round(line['y_start']).astype(int), 
                                           np.round(control_x).astype(int), np.round(control_y).astype(int), 
                                           np.round(end_x).astype(int),np.round(end_y).astype(int), 4)
            #plt.plot(curve_x, curve_y, alpha=0.7)
            #plt.xlim(0, box_size)
            #plt.ylim(0, box_size)
            
            df_linTemp = pd.DataFrame({
                'frame': line['frame'],
                'line_no': i,
                'line_intensity': line['line_intensity'],
                'x_positions': curve_x,
                'y_positions': curve_y
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
        '''
        if(noise=='yes'):
            maybe_blobs = np.random.randint(int(box_size/10))
            for l in range(maybe_blobs):
                if(np.random.uniform(0,1)>=0.9):
                    #print('blob!',i)
                    posx,posy = np.random.randint(0+4,box_size-4,2)
                    rc = disk((posx, posy), np.random.randint(1,5))
                    img_all[d,rc[0],rc[1]] = np.random.uniform(0.1,0.4)
        '''
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
            
            imgNoise[d] = skimage.util.random_noise(imgNoise[d], mode='gaussian', clip=True,mean=0, var=noiseBack).astype(dtype='float32')
            
            #normalise to 0-255, this is needed for conversion to tiff file
            imgNoise[d] = 255 * (imgNoise[d] - imgNoise[d].min()) / (imgNoise[d].max() - imgNoise[d].min()) 
        else:
            imgNoise[d] = imgf*255
            

   

    imgNoise = imgNoise.astype(np.uint8)

    tifffile.imwrite(path+filename + '.tiff', imgNoise, photometric='minisblack',imagej=True)

        
    return imgNoise
