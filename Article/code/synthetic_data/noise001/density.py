#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
import pandas as pd
import tifffile
import sys

overpath = 'define_this_path'


box_size=500

###############################################################################
#
# functions
#
###############################################################################

def density(filescsv,csvlist,box_size,len_files,imgSavePath):
    densAllImg=[]
    
    for kl in range(len(csvlist)):
        
        
        df_Ori = pd.read_csv(filescsv+csvlist[kl])
        
        # filter away all values outside our frame of view
        df_Ori = df_Ori[(df_Ori['x_positions'] >= 0) & (df_Ori['x_positions'] < box_size)]
        df_Ori = df_Ori[(df_Ori['y_positions'] >= 0) & (df_Ori['y_positions'] < box_size)]
        
        frames = np.unique(df_Ori['frame'])
        #density=np.zeros(len(frames))
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
            
        #densAll = np.append(densAll,density)
        densAllImg = np.append(densAllImg,densityImg)
        
        
    df_densAll = pd.DataFrame({
        'frame': np.arange(0,len_files),
        #'Density': densAll,
        'Density full image': densAllImg
    })
    
    return df_densAll

###############################################################################
#
# calculating density for all images with noise 0.01
#
###############################################################################

savecsv_Dens = overpath+'/noise001/lines/density/'

###############################################################################
#
# calculating density for 5 lines images
#
###############################################################################

imgSavePath5 = overpath+'/noise001/lines/lines_5/imgs/'
path_imgs5 = overpath+'/noise001/noise5/'
savepath5 = overpath+'/noise001/lines/lines_5/'

files5 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

df_densAll5 = density(filescsv=savepath5,csvlist=files5,box_size=500,len_files=501,imgSavePath = imgSavePath5)
df_densAll5.to_csv(savecsv_Dens + 'density5.csv', index=False) 
   
counts, bins = np.histogram(df_densAll5['Density full image'], bins=50)
# Normalize the histogram counts so that the sum is 1
counts = counts / counts.sum()
plt.figure(figsize=(8.27,5))
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel('Density full image')
plt.tight_layout()
#plt.savefig(path_imgs5+'figs/densityGraFT_fullImage.png')

###############################################################################
#
# calculating density for 10 lines images
#
###############################################################################

imgSavePath10 = overpath+'/noise001/lines/lines_10/imgs/'
path_imgs10 = overpath+'/noise001/noise10/'
savepath10 = overpath+'/noise001/lines/lines_10/'

files10 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

df_densAll10 = density(filescsv=savepath10,csvlist=files10,box_size=500,len_files=501,imgSavePath = imgSavePath10)
df_densAll10.to_csv(savecsv_Dens + 'density10.csv', index=False) 
   
counts, bins = np.histogram(df_densAll10['Density full image'], bins=50)
# Normalize the histogram counts so that the sum is 1
counts = counts / counts.sum()
plt.figure(figsize=(8.27,5))
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel('Density full image')
plt.tight_layout()
#plt.savefig(path_imgs5+'figs/densityGraFT_fullImage.png')

###############################################################################
#
# calculating density for 20 lines images
#
###############################################################################

imgSavePath20 = overpath+'/noise001/lines/lines_20/imgs/'
path_imgs20 = overpath+'/noise001/noise20/'
savepath20 = overpath+'/noise001/lines/lines_20/'

files20 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

df_densAll20 = density(filescsv=savepath20,csvlist=files20,box_size=500,len_files=501,imgSavePath = imgSavePath20)
df_densAll20.to_csv(savecsv_Dens + 'density20.csv', index=False) 
   
counts, bins = np.histogram(df_densAll20['Density full image'], bins=50)
# Normalize the histogram counts so that the sum is 1
counts = counts / counts.sum()
plt.figure(figsize=(8.27,5))
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel('Density full image')
plt.tight_layout()
#plt.savefig(path_imgs5+'figs/densityGraFT_fullImage.png')

###############################################################################
#
# calculating density for 30 lines images
#
###############################################################################

imgSavePath30 = overpath+'/noise001/lines/lines_30/imgs/'
path_imgs30 = overpath+'/noise001/noise30/'
savepath30 = overpath+'/noise001/lines/lines_30/'

files30 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

df_densAll30 = density(filescsv=savepath30,csvlist=files30,box_size=500,len_files=501,imgSavePath = imgSavePath30)
df_densAll30.to_csv(savecsv_Dens + 'density30.csv', index=False) 
   
counts, bins = np.histogram(df_densAll30['Density full image'], bins=50)
# Normalize the histogram counts so that the sum is 1
counts = counts / counts.sum()
plt.figure(figsize=(8.27,5))
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel('Density full image')
plt.tight_layout()
#plt.savefig(path_imgs5+'figs/densityGraFT_fullImage.png')

###############################################################################
#
# calculating density for 40 lines images
#
###############################################################################

imgSavePath40 = overpath+'/noise001/lines/lines_40/imgs/'
path_imgs40 = overpath+'/noise001/noise40/'
savepath40 = overpath+'/noise001/lines/lines_40/'

files40 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']

df_densAll40 = density(filescsv=savepath40,csvlist=files40,box_size=500,len_files=501,imgSavePath = imgSavePath40)
df_densAll40.to_csv(savecsv_Dens + 'density40.csv', index=False) 
   
counts, bins = np.histogram(df_densAll40['Density full image'], bins=50)
# Normalize the histogram counts so that the sum is 1
counts = counts / counts.sum()
plt.figure(figsize=(8.27,5))
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel('Density full image')
plt.tight_layout()
#plt.savefig(path_imgs5+'figs/densityGraFT_fullImage.png')