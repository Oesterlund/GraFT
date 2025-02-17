#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: isabella
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys

path="where_dataUtils_is"
sys.path.append(path)

import dataUtils

plt.close('all')

overpath = 'define_this_path'
saveOverpath = +overpath'/noise001/'


box_size=500
curvature_factor = 10
noiseBack = 0.01

###############################################################################
#
# create dataset with 5 lines
#
###############################################################################

savepathLines5 = overpath+'/noise001/lines/lines_5/'


num_lines = 5

df_many = pd.DataFrame()    
for m in range(501):
    print(m)
    df_1 = dataUtils.generate_filamentous_structure(box_size, num_lines, curvature_factor)  
    df_1['frame'] = m
    df_many = pd.concat([df_many, df_1], ignore_index=True)
    if((m%100==0) & (m!=0)):
        
        df_many.to_csv(savepathLines5 + '{0}_lines_intermediate.csv'.format(m), index=False)  
        df_many = pd.DataFrame()


    

savepath5 = saveOverpath + 'noise5/'
files5 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

for l in range(len(files5)):
    df_many = pd.read_csv(savepathLines5+files5[l])
    filename = re.sub(r'.csv', '', files5[l])
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath5,filename=filename)
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath5,filename=filename+'_nonoise',noise='no')
          
    
###############################################################################
#
# create dataset with 10 lines
#
###############################################################################

savepathLines10 = overpath+'/noise001/lines/lines_10/'

num_lines = 10

df_many = pd.DataFrame()    
for m in range(501):
    print(m)
    df_1 = dataUtils.generate_filamentous_structure(box_size, num_lines, curvature_factor)  
    df_1['frame'] = m
    df_many = pd.concat([df_many, df_1], ignore_index=True)
    if((m%100==0) & (m!=0)):
        
        df_many.to_csv(savepathLines10 + '{0}_lines_intermediate.csv'.format(m), index=False)  
        df_many = pd.DataFrame()


savepath10 = saveOverpath + 'noise10/'
files10 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']


for l in range(len(files10)):
    df_many = pd.read_csv(savepathLines10+files10[l])
    filename = re.sub(r'.csv', '', files10[l])
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath10,filename=filename)
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath10,filename=filename+'_nonoise',noise='no')
    
###############################################################################
#
# create dataset with 20 lines
#
###############################################################################

savepathLines20 = overpath+'/noise001/lines/lines_20/'

num_lines = 20

df_many = pd.DataFrame()    
for m in range(501):
    print(m)
    df_1 = dataUtils.generate_filamentous_structure(box_size, num_lines, curvature_factor)  
    df_1['frame'] = m
    df_many = pd.concat([df_many, df_1], ignore_index=True)
    if((m%100==0) & (m!=0)):
        
        df_many.to_csv(savepathLines20 + '{0}_lines_intermediate.csv'.format(m), index=False)  
        df_many = pd.DataFrame()


savepath20 = saveOverpath + 'noise20/'
files20 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']


for l in range(len(files20)):
    df_many = pd.read_csv(savepathLines20+files20[l])
    filename = re.sub(r'.csv', '', files20[l])
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath20,filename=filename)
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath20,filename=filename+'_nonoise',noise='no')


###############################################################################
#
# create dataset with 30 lines
#
###############################################################################

savepathLines30 = overpath+'/noise001/lines/lines_30/'

num_lines = 30

df_many = pd.DataFrame()    
for m in range(501):
    print(m)
    df_1 = dataUtils.generate_filamentous_structure(box_size, num_lines, curvature_factor)  
    df_1['frame'] = m
    df_many = pd.concat([df_many, df_1], ignore_index=True)
    if((m%100==0) & (m!=0)):
        
        df_many.to_csv(savepathLines30 + '{0}_lines_intermediate.csv'.format(m), index=False)  
        df_many = pd.DataFrame()


    

savepath30 = saveOverpath + 'noise30/'
files30 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

for l in range(len(files30)):
    df_many = pd.read_csv(savepathLines30+files30[l])
    filename = re.sub(r'.csv', '', files30[l])
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath30,filename=filename)
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath30,filename=filename+'_nonoise',noise='no')


###############################################################################
#
# create dataset with 40 lines
#
###############################################################################

savepathLines40 = overpath+'/noise001/lines/lines_40/'

num_lines = 40

df_many = pd.DataFrame()    
for m in range(501):
    print(m)
    df_1 = dataUtils.generate_filamentous_structure(box_size, num_lines, curvature_factor)  
    df_1['frame'] = m
    df_many = pd.concat([df_many, df_1], ignore_index=True)
    if((m%100==0) & (m!=0)):
        
        df_many.to_csv(savepathLines40 + '{0}_lines_intermediate.csv'.format(m), index=False)  
        df_many = pd.DataFrame()


savepath40 = saveOverpath + 'noise40/'
files40 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

for l in range(len(files40)):
    df_many = pd.read_csv(savepathLines40+files40[l])
    filename = re.sub(r'.csv', '', files40[l])
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath40,filename=filename)
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_many,noiseBlur=1,noiseBack=noiseBack,box_size=box_size,path=savepath40,filename=filename+'_nonoise',noise='no')
          