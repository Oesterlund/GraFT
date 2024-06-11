#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:29:36 2023

@author: pgf840
"""

import os
import numpy as np
import pandas as pd
import skimage
from skimage.filters import threshold_otsu
import scipy as sp
import networkx as nx
import math
from simplification.cutil import ( simplify_coords_vw)
from skimage.morphology import dilation, square
from scipy import ndimage, sparse
from collections import Counter
from scipy import stats
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import astropy.stats
import plotly.graph_objects as go
from astropy import units as u
from skspatial.objects import Line, Point
import pickle
import functools



###############################################################################
#
# create graph
#
###############################################################################

def segmentation_skeleton(image, sigma, small,thresh_top):
    '''
    Function for preprocessing and skeletonization of image-data.
    
    Parameters
    ----------
    image : array
        Image input.
    sigma : float
        Amount to perform gaussina spread with.
    small : float
        Removal of small coherent pixel groups.
    thresh_top : float
        lower bound of threshold  value.

    Returns
    -------
    ImF, imageCleaned, imH.

    '''
    # 1) gaussian filter
    imG=skimage.filters.gaussian(image,sigma)   
    # 2) frangi tubeness
    imF = skimage.filters.frangi(imG, sigmas=np.arange(1, 2, 0.1), scale_step=0.1, alpha=0.1, beta=2, gamma=15, black_ridges=False, mode='reflect', cval=0)
    # 3) morph grey closing with SE of disk on r=2
    #circle = skimage.morphology.selem.disk(2)
    #imC = skimage.morphology.grey.closing(imF, circle)
    # 4) CLAHE
    imE = skimage.exposure.equalize_adapthist(imF, kernel_size=None, clip_limit=0.01, nbins=256)
    # 5) median filter
    imM = sp.ndimage.median_filter(imE, size=(2,2),  mode='reflect', cval=0.0, origin=0)
    # 6) hysteresis
    thresh = threshold_otsu(imM)
    imH = skimage.filters.apply_hysteresis_threshold(imE, thresh*thresh_top, thresh)
    # 7) morph grey closing again
    #imC2 =skimage.morphology.grey.closing(imH*1, circle)
    # 8) skeletonize
    imS = skimage.morphology.skeletonize(imH > 0)
    # 9) remove small objects
    imageCleaned = skimage.morphology.remove_small_objects(imS, small, connectivity=2) > 0
    return(imF,imageCleaned,imH)

def node_initial(imageSkeleton):
    '''
    Creation of the original nodes marked on skeletonized image
    
    Parameters
    ----------
    imageSkeleton : array
        Skeletonised image.

    Returns
    -------
    Skeletonized image.

    '''

    imageFiltered = node_find(imageSkeleton)

    return (imageFiltered+imageSkeleton)

def project_edges(imE,eps,size):
    '''
    Function to add additional nodes to graph using VW algorithm

    Parameters
    ----------
    imE : array
        Skeletonized image with nodes marked.
    eps : float
        VW threshold.
    size : float
        limit value.

    Returns
    -------
    filtF2 : array
        Skeletonized image containing additional nodes.
    blank2 : array
        all addintioanl nodes marked on blanck image.

    '''
    imEc = imE.copy()
    #add padding
    imEc = np.pad(imEc, 1, 'constant')
    filt = imEc.copy()
    filtF = imEc.copy()
    
    (rows,cols) = np.nonzero((imEc>1)*1)
 
    M,N = filt.shape
    filtU = imEc.copy()
    blank = np.zeros((M,N))
    
    edgeval = 3
    
    for k in range(len(rows)):

        line_coords = []
        counter = 0
        stop2 = 0
        stop3 = 0
        stop4 = 0
        r = rows[k]
        c = cols[k]
 
        line_coords.append((r,c))
        
        filt = imEc.copy()
        filtU[r,c] = 0
        
        filt[r,c] = edgeval
        #filt[pos[edgesL[k][1]][1],pos[edgesL[k][1]][0]] = edgeval

        # start test on intial node entry to see if multiple tracings has to be done.
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        
        col_neighOri = col_neigh.astype('int')
        row_neighOri = row_neigh.astype('int')
        
        imageSection = filtU[row_neigh,col_neigh]
        
        imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)
        
        # need to have a check for if no value is over 1
        if((np.max(filtU)==1) and (labels4>0)):
            line_coords4 = line_coords.copy()
            # if this has happened, then there is a loop like structure left, so this tracing is different from rest
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')
            
            imageSection = filtU[row_neigh,col_neigh]
            
            imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)
            
            filtU4 = filtU.copy()
            filt4 = filt.copy()
            stop4 = 0
            filtU4[row_neigh,col_neigh] = (filtU4[row_neigh,col_neigh]==1)*1
            #imageLabeled4 = (imageLabeled4==(i+1))*1
            ind = np.where(imageLabeled4)
            index = 0
            if(len(ind[0])!=1):
                index = int(np.floor(len(ind)/2.))
            #move one
            r4 = (r+ind[0])[index]-1
            c4 = (c+ind[1])[index]-1
            line_coords4.append((r4,c4))
            
            filt4[r4,c4] = edgeval
            filtU4[r4,c4] = 0
            
            #check how it now looks
            (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))
        
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')
            
            imageSection = filtU4[row_neigh,col_neigh]
            imageLabeled4, labels4 = sp.ndimage.label(imageSection)
            
            while((labels4==1) and (stop4==0)):
                ind = np.where(imageLabeled4)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r4 = (r4+ind[0])[index]-1
                c4 = (c4+ind[1])[index]-1
                line_coords4.append((r4,c4))
                
                filt4[r4,c4] = edgeval
                filtU4[r4,c4] = 0
                
                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))
            
                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')
                
                imageSection = filtU4[row_neigh,col_neigh]
                imageLabeled4, labels4 = sp.ndimage.label(imageSection)
                
                if(np.max(imageSection)==0):
                    counter += 1
                    stop4 = 1
                    filtU[rows[k],cols[k]]=1
                    filtU = filtU - (filt4 == edgeval)*1
                    # do the rdp algorithm
                    #ind_stack = np.column_stack(np.where(edgeImg==edgeval))
                    rdpInd = simplify_coords_vw(line_coords4, eps)
                    #there is only one, so need to ad one in
                    for i in range(len(rdpInd)-1):
                        if(imEc[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])]==1 ):
                            filtF[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
                            blank[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
            
                                    
                    
            
        if(len(np.nonzero(imageSection)[0])>0):
            if(np.sum(imageSection)/len(np.nonzero(imageSection)[0])==2):
                #this means that we are only neighbours to other nodes
                stop4=1
            
        # i will run the amounts of runs as there are labels
        if(stop4==0):
            line_coords3 = line_coords.copy()
            for i in range(labels4):
                
                filtU3 = filtU.copy()
                filt3 = filt.copy()
                stop3 = 0
                filtU3[row_neighOri,col_neighOri] = filtU3[row_neighOri,col_neighOri]*(imageLabeled4==(i+1))*1 
                imageLabeled3 = (imageLabeled4==(i+1))*1
                ind = np.where(imageLabeled3)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r3 = (r+ind[0])[index]-1
                c3 = (c+ind[1])[index]-1
                line_coords3.append((r3,c3))
                
                filt3[r3,c3] = edgeval
                filtU3[r3,c3] = 0
                
                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))
            
                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')
                
                imageSection = filtU3[row_neigh,col_neigh]
                imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                
                
                while((labels3==1) and (stop3==0)):
                    ind = np.where(imageLabeled3)
                    index = 0
                    if(len(ind[0])!=1):
                        index = int(np.floor(len(ind)/2.))
                    #move one
                    r3 = (r3+ind[0])[index]-1
                    c3 = (c3+ind[1])[index]-1
                    line_coords3.append((r3,c3))
                    
                    filt3[r3,c3] = edgeval
                    filtU3[r3,c3] = 0
                    
                    #check how it now looks
                    (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))
                
                    col_neigh = col_neigh.astype('int')
                    row_neigh = row_neigh.astype('int')
                    
                    imageSection = filtU3[row_neigh,col_neigh]
                    imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                    # NEW
                    if(np.max(imageSection)==0):
                        #check if a node exist in imEc
                        if(np.max(imEc[row_neigh,col_neigh])==2):
                            #the loop ends here
                            # this is the stopper of this while loop for one trace only
                            stop3 = 1
                            stop2 = 1
                            counter += 1
                            
                            #filtU[rows[k],cols[k]]=1
                            filtU = filtU - (filt3 == edgeval)*1
                            
                            # do the VW algorithm
                            #ind_stack = np.column_stack(np.where(edgeImg==edgeval))
                            rdpInd = simplify_coords_vw(line_coords3, eps)
                            #if(len(rdpInd)>2):
                            for i in range(len(rdpInd)-1):
                                if(imEc[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])]==1 ):
                                    filtF[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
                                    blank[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
                        
                    elif((np.max(imageSection)>1)):
                        ind = np.where(imageSection==2)
                        index = 0
                        #move one
                        r3 = (r3+ind[0])[index]-1
                        c3 = (c3+ind[1])[index]-1
                        line_coords3.append((r3,c3))
                        
                        filt3[r3,c3] = edgeval
                        filtU3[r3,c3] = 0
                        
                        # this is the stopper of this while loop for one trace only
                        counter += 1
                        stop3 = 1
                        stop2 = 1
                        filtU[rows[k],cols[k]]=1
                        filtU = filtU - (filt3 == edgeval)*1
                        # do the VW algorithm
                        #ind_stack = np.column_stack(np.where(edgeImg==edgeval))
                        rdpInd = simplify_coords_vw(line_coords3, eps)  
                        #if(len(rdpInd)>2):
                        for i in range(len(rdpInd)-1):
                            if(imEc[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])]==1 ):
                                filtF[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
                                blank[int(rdpInd[i+1][0]),int(rdpInd[i+1][1])] = 2
    
    # if a structure appear with no premarked nodes, we will find it with this search
    structure=np.ones((3,3))
    labeled, nr_objects = ndimage.label(filtU,structure=structure) 
    while nr_objects:
        nr_objects -= 1
        if(np.sum((labeled==(nr_objects+1))*1)>size):
            line_coords = []
            indl = np.where(labeled*1==1)
            filtUl = (labeled*1).copy()
            filtU[r,c] = 0
            filt[r,c] = edgeval
            filtF[r,c] = 2 # add this as extra node
            blank[r,c] = 2 # add this as extra node
            r,c = indl[0][0],indl[1][0]
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        
            col_neighOri = col_neigh.astype('int')
            row_neighOri = row_neigh.astype('int')
            
            imageSection = filtUl[row_neigh,col_neigh]
            
            imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)
            
            line_coords3 = line_coords.copy()
            for i in range(labels4):
                
                filtU3 = filtU.copy()
                filt3 = filtU.copy()
                stop3 = 0
                filtU3[row_neighOri,col_neighOri] = filtU3[row_neighOri,col_neighOri]*(imageLabeled4==(i+1))*1 
                imageLabeled3 = (imageLabeled4==(i+1))*1
                ind = np.where(imageLabeled3)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r3 = (r+ind[0])[index]-1
                c3 = (c+ind[1])[index]-1
                line_coords3.append((r3,c3))
                
                filt3[r3,c3] = edgeval
                filtU3[r3,c3] = 0
                
                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))
            
                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')
                
                imageSection = filtU3[row_neigh,col_neigh]
                imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                
                
                while((labels3==1) and (stop3==0)):
                    ind = np.where(imageLabeled3)
                    index = 0
                    if(len(ind[0])!=1):
                        index = int(np.floor(len(ind)/2.))
                    #move one
                    r3 = (r3+ind[0])[index]-1
                    c3 = (c3+ind[1])[index]-1
                    line_coords3.append((r3,c3))
                    
                    filt3[r3,c3] = edgeval
                    filtU3[r3,c3] = 0
                    
                    #check how it now looks
                    (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))
                
                    col_neigh = col_neigh.astype('int')
                    row_neigh = row_neigh.astype('int')
                    
                    imageSection = filtU3[row_neigh,col_neigh]
                    imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                    # NEW
                    if(np.max(imageSection)==0):
                        #check if a node exist in imEc
                        if(np.max(imEc[row_neigh,col_neigh])==2):
                            #the loop ends here
                            # this is the stopper of this while loop for one trace only
                            stop3 = 1
                            counter += 1
                            
                            #filtU[rows[k],cols[k]]=1
                            filtU = filtU - (filt3 == edgeval)*1
                            
                            labeled, nr_objects = ndimage.label(filtU,structure=structure)
                            
                            # do the VW algorithm
                            #ind_stack = np.column_stack(np.where(edgeImg==edgeval))
                            rdpInd = simplify_coords_vw(line_coords3, eps)
                            #if(len(rdpInd)>2):
                            for sd in range(len(rdpInd)-1):
                                if(imEc[int(rdpInd[sd+1][0]),int(rdpInd[sd+1][1])]==1 ):
                                    filtF[int(rdpInd[sd+1][0]),int(rdpInd[sd+1][1])] = 2
                                    blank[int(rdpInd[sd+1][0]),int(rdpInd[sd+1][1])] = 2

    #remove padding
    filtF2 = filtF[1:-1, 1:-1]
    blank2 = blank[1:-1, 1:-1]

    return filtF2,blank2

def project_mask(imF):
    '''
    Create mask containing each edge marked with a unique value
    
    Parameters
    ----------
    imF : array
        Skeletonized image with marked nodes.

    Returns
    -------
    mask2 : array
        DESCRIPTION.
    index_list : list
        list of pixels for mask.

    '''
    # add padding
    img = np.pad(imF, 1, 'constant')
    
    filt = img.copy()
    
    mask = np.zeros((img.shape))
    
    (rows,cols) = np.nonzero((img>1)*1)
 
    M,N = filt.shape
    
    edgeval = 3
    count_mask = 4
    index_list = []
    
    for k in range(len(rows)):

        line_coords = []
        counter = 0
        stop3 = 0
        stop4 = 0
        r = rows[k]
        c = cols[k]
 
        line_coords.append((r,c))
        
        filt = img.copy()
        filtU = img.copy()
        filtU[r,c] = 0
        filt[r,c] = edgeval

        # start test on intial node entry to see if multiple tracings has to be done.
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        
        col_neighOri = col_neigh.astype('int')
        row_neighOri = row_neigh.astype('int')
        
        imageSection = filtU[row_neighOri,col_neighOri]
        
        imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)
        
        # need to have a check for if no value is over 1
        if((np.max(filtU)==1) and (labels4>0)):
            line_coords4 = line_coords.copy()
            # if this has happened, then there is a loop like structure left, so this tracing is different from rest
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')
            
            imageSection = filtU[row_neigh,col_neigh]
            
            imageLabeled4, labels4 = sp.ndimage.label((imageSection==1)*1)
            
            filtU4 = filtU.copy()
            filt4 = filt.copy()
            stop4 = 0
            filtU4[row_neigh,col_neigh] = (filtU4[row_neigh,col_neigh]==1)*1
            #imageLabeled4 = (imageLabeled4==(i+1))*1
            ind = np.where(imageLabeled4)
            index = 0
            if(len(ind[0])!=1):
                index = int(np.floor(len(ind)/2.))
            #move one
            r4 = (r+ind[0])[index]-1
            c4 = (c+ind[1])[index]-1
            line_coords4.append((r4,c4))
            
            filt4[r4,c4] = edgeval
            filtU4[r4,c4] = 0
            
            #check how it now looks
            (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))
        
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')
            
            imageSection = filtU4[row_neigh,col_neigh]
            imageLabeled4, labels4 = sp.ndimage.label(imageSection)
            
            while((labels4==1) and (stop4==0)):
                ind = np.where(imageLabeled4)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r4 = (r4+ind[0])[index]-1
                c4 = (c4+ind[1])[index]-1
                line_coords4.append((r4,c4))
                
                filt4[r4,c4] = edgeval
                filtU4[r4,c4] = 0
                
                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c4-1,c4,c4+1]), np.array([r4-1,r4,r4+1]))
            
                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')
                
                imageSection = filtU4[row_neigh,col_neigh]
                imageLabeled4, labels4 = sp.ndimage.label(imageSection)
                
                if(np.max(imageSection)==0):
                    counter += 1
                    stop4 = 1
                    filtU[rows[k],cols[k]]=1
                    filtU = filtU - (filt4 == edgeval)*1
                    
                    # mark out traced line in mask
                    node1 = line_coords4[0]
                    node2 = r4,c4
                    mask[node2] = 2
                    count_mask += 1
                    ks = 0
                    index_list.append((np.subtract(node1,1),np.subtract(node2,1),count_mask))
                    while(node2!=line_coords4[ks]):
                        mask[line_coords4[ks]] = count_mask   
                        ks += 1   
                                    
                    
            
        if(len(np.nonzero(imageSection)[0])>0):
            if(np.sum(imageSection)/len(np.nonzero(imageSection)[0])==2):
                #this means that we are only neighbours to other nodes
                stop4=1
            
        # i will run the amounts of runs as there are labels
        if(stop4==0):
            
            for i in range(labels4):
                line_coords3 = line_coords.copy()
                filtU3 = filtU.copy()
                filt3 = filt.copy()
                stop3 = 0
                filtU3[row_neighOri,col_neighOri] = filtU3[row_neighOri,col_neighOri]*(imageLabeled4==(i+1))*1 
                imageLabeled3 = (imageLabeled4==(i+1))*1
                ind = np.where(imageLabeled3)
                index = 0
                if(len(ind[0])!=1):
                    index = int(np.floor(len(ind)/2.))
                #move one
                r3 = (r+ind[0])[index]-1
                c3 = (c+ind[1])[index]-1
                line_coords3.append((r3,c3))
                
                filt3[r3,c3] = edgeval
                filtU3[r3,c3] = 0
                
                #check how it now looks
                (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))
            
                col_neigh = col_neigh.astype('int')
                row_neigh = row_neigh.astype('int')
                
                imageSection = filtU3[row_neigh,col_neigh]
                imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                
                fS=1
                
                while(((labels3==1) and (stop3==0)) or ((fS==1) and (labels3>0))):
                    fS=0
                    ind = np.where(imageLabeled3)
                    index = 0
                    if(len(ind[0])!=1):
                        index = int(np.floor(len(ind)/2.))
                    #move one
                    r3 = (r3+ind[0])[index]-1
                    c3 = (c3+ind[1])[index]-1
                    line_coords3.append((r3,c3))
                    
                    filt3[r3,c3] = edgeval
                    filtU3[r3,c3] = 0
                    
                    #check how it now looks
                    (col_neigh,row_neigh) = np.meshgrid(np.array([c3-1,c3,c3+1]), np.array([r3-1,r3,r3+1]))
                
                    col_neigh = col_neigh.astype('int')
                    row_neigh = row_neigh.astype('int')
                    
                    imageSection = filtU3[row_neigh,col_neigh]
                    imageLabeled3, labels3 = sp.ndimage.label(imageSection)
                    if(np.max(imageSection)==0):
                        #check if a node exist in imEc
                        if(np.max(img[row_neigh,col_neigh])==2):
                            #the loop ends here
                            # this is the stopper of this while loop for one trace only
                            counter += 1
                            stop3 = 1
                            node1 = line_coords3[0]
                            node2 = r3,c3
                            mask[node2] = 2
                            count_mask += 1
                            ks = 0
                            index_list.append((np.subtract(node1,1),np.subtract(node2,1),count_mask))
                            while(node2!=line_coords3[ks]):
                                mask[line_coords3[ks]] = count_mask   
                                ks += 1
                                        
                    elif(np.max(imageSection)>1):
                        ind = np.where(imageSection==2)
                        index = 0
                        #move one
                        r3 = (r3+ind[0])[index]-1
                        c3 = (c3+ind[1])[index]-1
                        line_coords3.append((r3,c3))
                        
                        filt3[r3,c3] = edgeval
                        filtU3[r3,c3] = 0
                        
                        # this is the stopper of this while loop for one trace only
                        counter += 1
                        stop3 = 1
                        # mark out traced line in mask
                        node1 = line_coords3[0]
                        node2 = r3,c3
                        mask[node2] = 2
                        count_mask += 1
                        ks = 0
                        index_list.append((np.subtract(node1,1),np.subtract(node2,1),count_mask))
                        while(node2!=line_coords3[ks]):
                            mask[line_coords3[ks]] = count_mask   
                            ks += 1

                    
    #remove padding
    mask2 = mask[1:-1, 1:-1]
    return mask2,index_list

def condense_mask(index_list,imageNodeCondense,mask,size):
    '''
    Create list containing node positions

    Parameters
    ----------
    index_list : TYPE
        DESCRIPTION.
    imageNodeCondense : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    df_new : TYPE
        DESCRIPTION.

    '''
    if(index_list==[]):
        df_new = []
        
    else:
        len_index = np.zeros(len(index_list))
        correct_pos1 = [0]*len(index_list)
        correct_pos2 = [0]*len(index_list)
        nodenum1 = np.zeros(len(index_list))
        nodenum2 = np.zeros(len(index_list))
        posx,posy = np.nonzero((imageNodeCondense>0)*1)
        index1 = np.zeros(len(posx))
        index2 = np.zeros(len(posx))
        for i in range(len(index_list)):
            #if smaller than 10 in length, then remove this one
            len_index[i] = np.sqrt((index_list[i][0][0] - index_list[i][1][0] )**2 + (index_list[i][0][1] - index_list[i][1][1] )**2 )
            for j in range(len(posx)):
                index1[j] = np.sqrt((index_list[i][0][0] - posx[j] )**2 + (index_list[i][0][1] - posy[j] )**2 )
                index2[j] = np.sqrt((index_list[i][1][0] - posx[j] )**2 + (index_list[i][1][1] - posy[j] )**2 )
            argm1 = np.argmin(index1)
            argm2 = np.argmin(index2)
            correct_pos1[i] = list((posx[argm1],posy[argm1]))
            nodenum1[i] = argm1
            correct_pos2[i] = list((posx[argm2],posy[argm2]))
            nodenum2[i] = argm2
    
        index_pd = pd.DataFrame(index_list)
        index_pd[3] = len_index
        index_pd[4] = correct_pos1
        index_pd[5] = nodenum1
        index_pd[6] = correct_pos2
        index_pd[7] = nodenum2
        #df = index_pd[index_pd[3] > size]
        df= index_pd
        
        df = df[df[2].isin(np.unique(mask))]
        df = df.reset_index(drop=True)
        vals = np.unique(df[2])
        delete_row= []
        ss = df.duplicated(subset=[5,7], keep=False)
        index_ss = ss[ss].index
        delete_df = []
        for i in index_ss:
            smallM = []
            indexM = []
            df_sub = df.loc[(df[5] == df[5].iloc[i]) & (df[7]==df[7].iloc[i])]
            #find smallest value
            for l in range(len(df_sub)):
                smallM.append(np.sum(mask==df_sub[2].iloc[l]))
                indexM.append(df_sub[2].iloc[l])
            argM = np.argmin(smallM)
            delete_df.append(df_sub[df_sub[2]==indexM[argM]].index[0])
        delI = np.unique(delete_df)
        df = df.drop(delI)   
        df = df.reset_index(drop=True)    
    
        df = df.drop(delete_row)
        df = df.reset_index(drop=True)
        index_remove = list(np.where(df[4]==df[6])[0])
        df = df.drop(index_remove,axis='index')   
        df = df.reset_index(drop=True)
        df_new = df.rename(columns={0: "old pos1", 1: "old pos2", 2: "map value", 3: "distance between pos", 4: "new pos1", 5: "node pos1", 6: "new pos2", 7: "node pos2"})
    return df_new

def node_condense(imageFiltered,imageSkeleton,kernel):
    '''
    Condensation of nodes based on kernel size set by user.

    Parameters
    ----------
    imageFiltered : array
        Skeletonized image containing nodes and the added nodes from VW algorithm.
    imageSkeleton : array
        Original skeletonized image with nodes marked.
    kernel : array
        kernel to filter on.

    Returns
    -------
    Skeletonized image with nodes defined and merged together based on distance.

    '''
    imageLabeled, labels = sp.ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    #need to have rolling window to condense nodes together
    imgSL = imageLabeled+imageSkeleton

    half = int(len(kernel)/2)
    M,N = imgSL.shape
    for l in range(half,M-half):
        for k in range(half,N-half):

            small = imgSL[l-half:l+half,k-half:k+half]
            # get all pixel location for vals higher than 1
            if((np.sum((small>1)*1)>2)):
                location=np.argwhere(small > 1)
                #if any endpoints,remove those.
                for z in range(len(location)):
                    coord1 = np.array([location[z,0],location[z,1]])
                    if(np.sum((imgSL[l-half+coord1[0]-1:l-half+coord1[0]+2,k-half+coord1[1]-1:k-half+coord1[1]+2]>0)*1)==2):
                        #this is an endpoint, remove it
                        imgSL[l-half+coord1[0],k-half+coord1[1]] = 1
                    small = imgSL[l-half:l+half,k-half:k+half]
                location=np.argwhere(small > 1)
                for i in range(len(location)):
                    if(i != (np.floor(len(location)/2))):
                        #this is the middle one , need to keep it
                        coord1 = np.array([location[i,0],location[i,1]])
                        imgSL[l-half+coord1[0],k-half+coord1[1]] = 1
            
            # there are two different points close to each other
            elif(np.sum((small>1)*1)==2):

                location=np.argwhere(small > 1)
                    
                # test that both points are not endnodes
                coord1 = np.array([location[0,0],location[0,1]])
                coord2 = np.array([location[1,0],location[1,1]])
                
                node1conn = np.sum((imgSL[l-half+coord1[0]-1:l-half+coord1[0]+2,k-half+coord1[1]-1:k-half+coord1[1]+2]>0)*1)
                node2conn = np.sum((imgSL[l-half+coord2[0]-1:l-half+coord2[0]+2,k-half+coord2[1]-1:k-half+coord2[1]+2]>0)*1)
                if((node1conn>2) & (node2conn>2)):
                    # nod end nodes
                    y_min = min(l-half+location[0][0],l-half+location[1][0])
                    y_max = max(l-half+location[0][0],l-half+location[1][0])
                    x_min = min(k-half+location[0][1], k-half+location[1][1])
                    x_max = max(k-half+location[0][1], k-half+location[1][1])
                    zoom = (imgSL[y_min:y_max+1,x_min:x_max+1]>0)*1
                    conV = skimage.measure.label(zoom, connectivity=2)
                    if(np.max(conV)==1):
                        com = ndimage.center_of_mass(conV)
                       
                        if(coord1[1]<coord2[1]):
                            coordinateC = math.ceil(com[0])+coord1[0],math.ceil(com[1])+coord1[1]
                            coordinateF = math.floor(com[0])+coord1[0],math.floor(com[1])+coord1[1]
                        else:
                            coordinateC = math.ceil(com[0])+coord1[0],-math.ceil(com[1])+coord1[1]
                            coordinateF = math.floor(com[0])+coord1[0],-math.floor(com[1])+coord1[1]
                        
                        # test if the coordinate has value different from 0
                        if((imgSL[l-half+coordinateC[0],k-half+coordinateC[1]])>0):
                            #contnue set value to one of the other values
                            val = imgSL[l-half+coord1[0],k-half+coord1[1]]
                            imgSL[l-half+location[0][0],k-half+location[0][1]] = 1
                            imgSL[l-half+location[1][0],k-half+location[1][1]] = 1
                            imgSL[l-half+coordinateC[0],k-half+coordinateC[1]] = val
                            
                        elif((imgSL[l-half+coordinateF[0],k-half+coordinateF[1]])>0):
                            val = imgSL[l-half+coord1[0],k-half+coord1[1]]
                            imgSL[l-half+location[0][0],k-half+location[0][1]] = 1
                            imgSL[l-half+location[1][0],k-half+location[1][1]] = 1
                            imgSL[l-half+coordinateF[0],k-half+coordinateF[1]] = val
                    elif(np.max(conV)==2):
                         # need to check that the two nodes are connected
                        arr,num = skimage.measure.label((small>0)*1,return_num=True, connectivity=2)
                        #if num = 2, then they are not together- meaning that its two seperated branches. if num=1 then together
                        if(num==1):
                            #remove end node
                            imgSL[l-half+coord2[0],k-half+coord2[1]] = 1
                else:
                    # need to check that the two nodes are connected
                    arr,num = skimage.measure.label((small>0)*1,return_num=True, connectivity=2)
                    #if num = 2, then they are not together- meaning that its two seperated branches. if num=1 then together
                    if(num==1):
                    #remove end node
                        if(node1conn==2):
                            imgSL[l-half+coord1[0],k-half+coord1[1]] = 1
                        if(node2conn==2):
                            imgSL[l-half+coord2[0],k-half+coord2[1]] = 1
    return (imgSL-imageSkeleton)  

def node_graph(imE,imA,size,eps):
    '''
    Function to collect other function and create annotated image and node positions together with helper outputs

    Parameters
    ----------
    imE : array
        Skeletonized image with nodes marked.
    imA : TYPE
        Skeletonized image.
    size : float
        size to merge nodes.
    eps : float
        Treshold for VW algorithm.

    Returns
    -------
    imageAnnotated : array
        Skeletonized image with nodes.
    imgBlR : array
        mage with added nodes.
    mask : array
        mask of edges.
    df_pos : list
        list of node positions.

    '''
    imF,imgBl = project_edges(imE,eps,size)
    mask,index_list = project_mask(imF)
    ones = np.ones((3, 3))
    # TODO FIXME: remove utilsF_performance after performance profiling
    # ~ imageNodeCondense = node_condense(imF-imA,imA, np.ones((size, size)))
    from graft import utilsF_performance
    imageNodeCondense = utilsF_performance.node_condense_11(imF-imA,imA, np.ones((size, size)))
    imgInt = dilation((imE>1)*1, square(size))
    imgBlR =(((imgBl>0)*1 - imgInt)>0)*1
    df_pos = condense_mask(index_list,imageNodeCondense,mask,size)
    imgReLab, labels = sp.ndimage.label(imageNodeCondense, structure=ones)
    imageAnnotated = imgReLab+imA
    return imageAnnotated,imgBlR,mask,df_pos

def node_find(imageSkeleton):
    '''
    Locate and mark position of nodes for the skeletonized image

    Parameters
    ----------
    imageSkeleton : array
        Original skeletonized image.

    Returns
    -------
    imageNodes : array
        Skeletonized image with nodes.

    '''
    
    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(imageSkeleton)
    imageNodes=np.zeros(np.shape(imageSkeleton))
    
    M,N = imageSkeleton.shape
    # For each non-zero pixel...
    for (r,c) in zip(rows,cols):
        imageSkeleton[r-1:r+2,c-1:c+2]
        # Extract an 8-connected neighbourhood
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')
        if ((r+1)<M and (c+1)<N):
            imageSection = imageSkeleton[row_neigh,col_neigh]
        else:
            imageSection = np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]])  
        #remove the center value to label adjacent values to it
        imageSection[1,1] = 0
        imageLabeled, labels = sp.ndimage.label(imageSection)
            
        if ((labels != 0 and labels != 2) or ( np.sum(imageSection*1)>=4)):
            imageNodes[r,c]=1
    return imageNodes

def edge_len(mask, df_pos,m):
    '''
    Calculate the length of each edge

    Parameters
    ----------
    mask : array
        mask of edges.
    df_pos : TYPE
        list of node positions.
    m : TYPE
        which edge is indexed in the mask array.

    Returns
    -------
    dist : float
        The distance of the edge.

    '''
    # calculate length for one edge
    edge_img = (mask==df_pos['map value'][m])*1
    (rows,cols) = np.nonzero(edge_img)
    dist = 1
    for (r,c) in zip(rows[:-1],cols[:-1]):
        if((edge_img[r,c+1]==1) or (edge_img[r,c-1]==1) or (edge_img[r+1,c]==1) or (edge_img[r-1,c]==1)):
            dist +=1
            edge_img[r-1:r+2,c-1:c+2]=0
        else:
            dist+= np.sqrt(2)
            edge_img[r-1:r+2,c-1:c+2]=0
    return dist

def make_graph_mask(imageAnnotated, imG, mask, df_pos):
    '''
    

    Parameters
    ----------
    imageAnnotated : array
        DESCRIPTION.
    imG : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.
    df_pos : TYPE
        DESCRIPTION.

    Returns
    -------
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.

    '''
    pos = np.transpose(np.where(imageAnnotated > 1))[:, ::-1]
    nodeNumber = imageAnnotated.max() - 1
    graph = nx.empty_graph(nodeNumber, nx.MultiGraph())
    for m in range(len(df_pos)):
        node1,node2 = imageAnnotated[df_pos["new pos1"][m][0],df_pos["new pos1"][m][1]]-2, imageAnnotated[df_pos["new pos2"][m][0],df_pos["new pos2"][m][1]]-2

        #filamentLengthSum = np.sum((mask==df_pos['map value'][m])*1)
        filamentIntensitySum = np.sum(imG*(mask==df_pos['map value'][m])*1)
        nodeDistance = np.sqrt((df_pos['new pos1'][m][0] - df_pos['new pos2'][m][0])**2 + (df_pos['new pos1'][m][1] - df_pos['new pos2'][m][1])**2 )
        filamentLengthSum = max(nodeDistance, edge_len(mask, df_pos,m))
        #node1,node2 = df_pos1["node pos1"][m], df_pos1["node pos2"][m]
        minimumEdgeWeight = max(1e-9, filamentIntensitySum)
        edgeCapacity = 1.0 * minimumEdgeWeight / filamentLengthSum
        edgeLength = 1.0 * filamentLengthSum / minimumEdgeWeight
        edgeConnectivity  = 0
        graph.add_edge(node1, node2, edist=nodeDistance, fdist=filamentLengthSum, weight=minimumEdgeWeight, capa=edgeCapacity, lgth=edgeLength, conn=edgeConnectivity)
    return graph,pos 

def unify_graph(graph):
    '''
    Project graph to simple graph

    Parameters
    ----------
    graph : nx graph
        graph.

    Returns
    -------
    simpleGraph : nx simple graph
        graph containing the same values, now converted.

    '''
    simpleGraph = nx.empty_graph(graph.number_of_nodes())
    for node1, node2, property in graph.edges(data=True):
        edist = property['edist']
        fdist = property['fdist']
        weight = property['weight']
        capa = property['capa']
        lgth = property['lgth']
        conn = property['conn']
        if simpleGraph.has_edge(node1, node2):
            simpleGraph[node1][node2]['capa'] += capa
            if(simpleGraph[node1][node2]['lgth'] > lgth):
                simpleGraph[node1][node2]['lgth'] = lgth
        else:
            simpleGraph.add_edge(node1, node2, edist=edist, fdist=fdist, weight=weight, capa=capa, lgth=lgth, conn=conn)
    return simpleGraph

def angle_between_edges(node1,node2,pos):
    '''
    Takes nodes from the linegraph as input, so nodes represents edges.
    From this can calculate the angle created between two edges

    Parameters
    ----------
    node1 : float
        first node, which represents edge.
    node2 : float
        Second node, which represents edge.
    pos : list
        list of position of nodes.

    Returns
    -------
    angle_deg : float
        The angle between two edges.

    '''
  
    same = np.intersect1d(node1,node2)
    index1 = np.where(node1==same)
    index2 = np.where(node2==same)
    
    if((index1[0]==1) and (index2[0]==1)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[0]]
        edgepos21 = pos[node2[1]]
    
    elif((index1[0]==0) and (index2[0]==1)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[1]]
        edgepos21 = pos[node2[0]]
    
    elif((index1[0]==0) and (index2[0]==0)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[0]]
        edgepos21 = pos[node2[1]]
        
    elif((index1[0]==1) and (index2[0]==0)):
        edgepos10 = pos[node1[0]]
        edgepos11 = pos[node1[1]]
        edgepos20 = pos[node2[1]]
        edgepos21 = pos[node2[0]]
       
    
    vec1 = (edgepos11[0]-edgepos10[0], edgepos11[1]-edgepos10[1])
    vec2 = (edgepos21[0]-edgepos20[0], edgepos21[1]-edgepos20[1])
    unit_vector_1 = vec1 / np. linalg. norm(vec1)
    unit_vector_2 = vec2 / np. linalg. norm(vec2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if(dot_product>1):
        dot_product=1
    elif(dot_product<-1):
        dot_product=-1
    angle = np.arccos(dot_product)
    
    angle_deg = angle*180./np.pi
    return angle_deg

def lG_edgeVal(lg1,graph1,pos):
    '''
    takes the linegraph and populate with angles, as well as marking all dangling nodes

    Parameters
    ----------
    lg1 : nx graph
        linegraph.
    graph1 : nx graph
        Original graph.
    pos : list
        List of positions of nodes.

    Returns
    -------
    lg1 : nx graph
        Linegraph populated with angle values of the nodes aka edges of the original graph.

    '''

    for node1, node2, property in lg1.edges(data=True):
        # calculate the angle between these two edges from the original graph
        # this is done by the positions in image
        lg1[node1][node2]['angle'] = angle_between_edges(node1,node2,pos)

        lg1[node1][node2]['intensity'] = np.abs(graph1[node1[0]][node1[1]]['capa'] - graph1[node2[0]][node2[1]]['capa'])/min(graph1[node1[0]][node1[1]]['capa'], graph1[node2[0]][node2[1]]['capa'])
        lg1.nodes[node1]['dangling'] =  graph1[node1[0]][node1[1]]['dangling']
        lg1.nodes[node2]['dangling'] =  graph1[node2[0]][node2[1]]['dangling']
    return lg1
  
def dangling_edges(graph1):
    '''
    mark all edges as either dangling (1) or not (0)

    Parameters
    ----------
    graph1 : nx graph
        Original graph.

    Returns
    -------
    The graph with marked dangling edges.

    '''
    # mark all edges with dangling = 0
    for node1, node2, property in graph1.edges(data=True):
        graph1[node1][node2]['dangling'] = 0
        graph1[node1][node2]['filament dangling'] = 0
    
        
    node_degree1_list = [k for k,v in nx.degree(graph1) if ((v == 1) or (v == 3))]
    
    for node1,node2 in graph1.edges(node_degree1_list):
        graph1[node1][node2]['dangling'] = 1
    return(graph1)


def creategraph(image,size,eps,thresh_top,sigma,small):
    '''
    Final function to create the graph, takes in image and all parameters needed.

    Parameters
    ----------
    image : array
        original image.
    size : float
        merge radius.
    eps : float
        Treshold for VW algorithm.
    thresh_top : float
        DESCRIPTION.
    sigma : float
        value for gaussian spread.
    small : float
        value for removal of coherent pixels.

    Returns
    -------
    gBuC : TYPE
        DESCRIPTION.
    pos : list
        list of node x,y positions.
    imA : array
        Skeletonized image.
    imgAF : array
        Image with node positions.
    imgBl : array
        Image with added nodes from VW algorithm.
    imF : array
        Image tube henancced and binarized.
    mask : array
        Image containing unique values for each edge.
    df_pos : list
        list of node positions, old mapped to new together with map value.

    '''

    imI = 255.0 * (image - image.min()) / (image.max() - image.min())  
    imG=imI 

    imR,imA,imF=segmentation_skeleton(imG,sigma,small,thresh_top)
    imG = np.pad(imG, 1, 'constant')
    imF = np.pad(imF, 1, 'constant')
    imA = np.pad(imA, 1, 'constant')
    imR = np.pad(imR, 1, 'constant')
    imA=imA*1
    
    imE = node_initial(imA>0)     

    imgAF,imgBl,mask,df_pos = node_graph(imE,imA,size,eps)
    
    if(len(df_pos)==0):
        print('the image does not contain filamentous structures, or parameters need tuning.')
        gBuC=0
        pos=0
        imA=0
        imgAF=0
        imgBl=0
        imF=0
        mask=0
        df_pos=0
    else:
        
        gBo,pos = make_graph_mask(imgAF,imG,mask,df_pos)
        
        gBu = unify_graph(gBo) 
        
        gBuC = test_connectivity(gBu)
        
    return gBuC,pos,imA,imgAF,imgBl,imF,mask,df_pos


###############################################################################
#
# tracing step
#
###############################################################################

def test_connectivity(graph):
    '''
    function to remove if contain single node as connected component

    Parameters
    ----------
    graph : nx graph
        raw graph.

    Returns
    -------
    graph : nx graph
        filtered graph.

    '''
    c_list = [list(c) for c in nx.connected_components(graph)]
    for i in c_list:
        if(len(i)==1):
            print(i)
            graph.remove_node(i[0])
    return graph


def all_angles(lg,pos):
    '''
    takes the linegraph and nodepositions in and reated a dataframe with nodepos and belonging angle

    Parameters
    ----------
    lg : array
        linegraph of graph.
    pos : list
        node positions.

    Returns
    -------
    df_new : list
        dataframe of nodepositions together with angle.

    '''
    all_anglesl = [list(c) for c in lg.edges(data='angle')]
    pd_al = pd.DataFrame(all_anglesl)
    df_new = pd_al.rename(columns={0: "pos1", 1: "pos2", 2: "angle"})
    return df_new

def min_angle(graph,imgBl,pos,angle):
    '''
    calculate minimum angle to use in constrained DFS

    Parameters
    ----------
    graph : nx graph
        input graph.
    imgBl : array
        image with the added nodes from VW algorithm.
    pos : list
        list of node positions.
    angle : float
        user input of angle.

    Returns
    -------
    min_angle_val : float
        Minimum angle allowed for the constrained DFS.

    '''
    graphD = dangling_edges(graph)
    lgG = nx.line_graph(graph)
    lg1 = lG_edgeVal(lgG,graphD,pos)

    # define angle list, find all coordinates in blank image and then find the 
    # overlap between pos and these. then find the angles between these coordinates
    # return this smallest value
    angles_list = [angle]
    (rows,cols) = np.nonzero((imgBl>0)*1)
    posBl = np.vstack((cols,rows)).T

    ind_pos = np.zeros(len(posBl))
    ind_posD = []
    for i in range(len(posBl)):
        ind_pos[i] = np.argmin(abs(posBl[i][0]-pos[:,0])+abs(posBl[i][1]-pos[:,1]))
        if(abs(posBl[i][0]-pos[int(ind_pos[i]),0])+abs(posBl[i][1]-pos[int(ind_pos[i]),1])>1):
            ind_posD.append(i)
    ind_posN = np.delete(ind_pos,ind_posD)
        
    angles = [list(c) for c in lg1.edges(data='angle')]
    
    for j in range(len(ind_posN)):
        for i in range(len(angles)):
            if((ind_posN[j] in angles[i][0:1][0]) and (ind_posN[j] in angles[i][1:2][0])):
                np.asarray(angles[i][0:1]).flatten() 
                angles_list.append(angles[i][2])
    min_angle_val = min(angles_list) - 1
    while(min_angle_val < 90):
        angles_list.remove(min(angles_list))
        min_angle_val = min(angles_list) - 1
    return min_angle_val


def dfs_constrained(graph_s,lgG_V,imgBl,pos,angle,overlap_allowed):
    '''
    constrained DFS function, to define and trace individual filaments

    Parameters
    ----------
    graph_s : nx graph
        graph from image data.
    lgG_V : nx graph
        linegraph from graph from image data.
    imgBl : array
        image with the added nodes from VW algorithm.
    pos : list
        list of node positions.
    angle : float
        minimum angle allowed.
    overlap_allowed : float
        minimum overlap allowed.

    Returns
    -------
    graphTagF : nx graph
        graph with tagged filament values.

    '''

    graphTag = graph_s.copy()
    graphTag = dangling_edges(graphTag)
    c_list = [list(c) for c in nx.connected_components(graph_s)]
    tag = 0
    to_add = []
    while c_list:

        c = c_list[0]
        c_list.pop(0)
        G = graph_s.subgraph(sorted(c)).copy()
        angle_min_val = min_angle(G,imgBl,pos,angle)
        al_pd = all_angles(lgG_V,pos)
        
        all_stacks = []
        path_full = []
        path_keep =[]
        
        dangling_nodeS = [k for k,v in nx.degree(G) if ((v == 1))]
        dangling_nodeE = [k for k,v in nx.degree(G) if ((v == 3))]
        dangling_node4 = [k for k,v in nx.degree(G) if ((v == 4))]
        
        dangling_nodeS.extend(dangling_nodeE)
        dangling_nodeS.extend(dangling_node4)
        dangling_nodeSC = dangling_nodeS.copy()

        df_I = pd.DataFrame(graph_s.edges(data='capa'))
        df_I[3] = np.asarray(list(graph_s.edges(data='fdist')))[:,2]
        df_I = df_I.rename(columns={0: "node1", 1: "node2", 2: "capa", 3: "fdist"})
        # need to add a check in case the smallest node value is not node1 from node2
        for i in range(len(df_I)):
            if( df_I['node1'].iloc[[i]].values[0] > df_I['node2'].iloc[[i]].values[0] ):
                #switch them around
                nod1 = df_I['node1'].iloc[[i]].values[0].copy()
                nod2 = df_I['node2'].iloc[[i]].values[0].copy()
                df_I.loc[i,'node1'] = nod2
                df_I.loc[i,'node2'] = nod1

        while dangling_nodeSC:
            dangling_del = []
            # edges for components with source
            source = dangling_nodeSC[0]
            dangling_nodeSC.pop(0)
            nodes = [source]
            visited = set()
            depth_limit = len(G)
            for start in nodes:
                if start in visited:
                    continue
                visited.add(start)
                stack = [(start, depth_limit, iter(G[start]))]
                while stack:
                    parent, depth_now, children = stack[-1]
                    try:
                        child = next(children)
                        if child not in visited:
                            #test that this path is allowed
                            if(len(path_keep)>=1):
                                entry = al_pd['angle'].loc[((al_pd['pos1'] == (min(parent,child),max(parent,child))) | (al_pd['pos1'] == path_keep[-1])) & ((al_pd['pos2'] == (min(parent,child),max(parent,child))) | (al_pd['pos2'] == path_keep[-1]))]
                                if(entry.size>0):
                                    if(entry.values>= angle_min_val): 
                                        #yield parent, child
                                        path_keep.append((min(parent,child),max(parent,child)))
                                        visited.add(child)
                                        if depth_now > 1:
                                            stack.append((child, depth_now - 1, iter(G[child])))
                                        #check for dangling nodes
                                        if(child in dangling_nodeS):
                                            #append full list
                                            all_stacks.append(stack.copy())
                                            path_full.append(path_keep.copy())
                                            dangling_del.append(child)
                                    if(entry.values < angle_min_val):
                                        #entry is too small
                                        
                                        if((len(path_full)==0)):
                                            if(nx.degree(G,parent) in {2,4}):
                                                dangling_del.append(child)
                                                path_full.append(path_keep.copy())

                                        
                                        elif((path_full[-1]!=path_keep) and (nx.degree(G,parent) in {2,4})):
                                            path_full.append(path_keep.copy())
                                            dangling_del.append(child)
                
                            else:
                                #this is the first one
                                #yield parent, child
                                path_keep.append((min(parent,child),max(parent,child)))
                                visited.add(child)
                                if depth_now > 1:
                                    stack.append((child, depth_now - 1, iter(G[child])))
                                #check for dangling nodes
                                if(child in dangling_nodeS):
                                    #append full list
                                    all_stacks.append(stack.copy())
                                    path_full.append(path_keep.copy())
                                    dangling_del.append(child)
                                
                                    
                                    
                        elif(((child in np.asarray([item for sublist in path_keep for item in sublist])) and (child not in path_keep[-1])) and ((child in dangling_nodeE) or (child in dangling_node4)) and (1 != path_keep.count((min(child,parent),max(child,parent))) )):
                            # add to path
                            path_keep.append((min(parent,child),max(parent,child)))
                            stack.append((child, depth_now - 1, iter(G[child])))
                            # save to path_full
                            all_stacks.append(stack.copy())
                            
                            if((child in dangling_nodeE)):
                                path_keep1 = list(set(path_keep) - set(path_full[-1]))
                                path_full.append(path_keep1)
                            path_full.append(path_keep.copy())
                            dangling_del.append(child)
    
                    except StopIteration:

                        stack.pop()
                        if(len(path_keep)>0):
                            path_keep.pop()

        # add all individual edges in as possible paths.
        path_full.extend([element] for element in list(graph_s.edges(sorted(c))))
        #first stack is done - calculate and choose best 
        len_path = np.zeros(len(path_full))
    
        for i in range(len(path_full)):
            path_c = path_full[i]
            iLl = np.zeros(len(path_c))
            for k in range(len(path_c)):
                iLl[k] = df_I['fdist'][(df_I['node1'] == path_c[k][0]) & (df_I['node2'] == path_c[k][1]) ]

            len_fil = np.sum(iLl)
            len_path[i] = len_fil
            #best one has index
            
        idx = np.argsort(len_path)
        path_keep = []
        
        full_len_path = len_path.copy()
        
        path_keep.append(path_full[idx[-1]])
        
        flat_list = np.asarray([item for sublist in path_keep for item in sublist])
        G_list = np.asarray(list(graph_s.edges(sorted(c))))

        not_covered = G_list[~(G_list[:, None] == flat_list).all(-1).any(-1)]
        
        while len(not_covered)>0:
            len_path = np.zeros(len(path_full))
            for i in range(len(path_full)):
                path_c = path_full[i]
                iLl = np.zeros(len(path_c))
                len_fil=0
                for k in range(len(path_c)):
                    if(np.isin(np.asarray(path_c[k]),np.asarray(not_covered)).all()):
                        iLl[k] = df_I['fdist'][(df_I['node1'] == path_c[k][0]) & (df_I['node2'] == path_c[k][1]) ]
                len_fil = np.sum(iLl)
                len_path[i] = len_fil
            
            idx = np.argsort(len_path)
            
            flat_list = [item for sublist in path_keep for item in sublist]
            switch=1
            #check that the end or start node is not in already chosen paths
            while (switch):
                if(((np.sum(np.isin(np.asarray(flat_list),np.asarray(path_full[idx[-1]])),axis=1)==2).any())==False):
                    switch=0
                    
                elif(((np.sum(np.isin(np.asarray(flat_list),np.asarray(path_full[idx[-1]][0])),axis=1)==2).any()) or ((np.sum(np.isin(np.asarray(flat_list),np.asarray(path_full[idx[-1]][-1])),axis=1)==2).any())):
                    # end/start edge overlap
                    idx = np.delete(idx,-1)
                else:
                    #other type of overlap
                    #calculate length of overlap
                    overlap_ind = np.asarray(np.sum(np.isin(np.asarray(flat_list),np.asarray(path_full[idx[-1]])),axis=1)==2).nonzero()[0]
                    overlap = [flat_list[i] for i in overlap_ind]
                    iLo=np.zeros(len(overlap))
                    for k in range(len(overlap)):
                        iLo[k] = df_I['fdist'][(df_I['node1'] == overlap[k][0]) & (df_I['node2'] == overlap[k][1]) ]
                    len_overlap = np.sum(iLo)
                    if(len_overlap > full_len_path[idx[-1]]/overlap_allowed):
                           #too large overlap, must remove
                           idx = np.delete(idx,-1)
                    else:
                        switch=0
            
            path_keep.append(path_full[idx[-1]])
            
            #update length
            flat_list = np.asarray([item for sublist in path_keep for item in sublist])
            G_list = np.asarray(list(graph_s.edges(sorted(c))))
            
            
            not_covered = G_list[~(G_list[:, None] == flat_list).all(-1).any(-1)]

        #now we hace removed all paths that are overlapping. now we need to check the best one against all others, and remove if too much
        for l in range(len(path_keep)):
            edges_to_remove = path_keep[l]
            overlap = []
            for i in range(l+1,len(path_keep)):
                
                if((np.sum(np.isin(np.asarray(path_keep[l]),np.asarray(path_keep[i])),axis=1)==2).any()):
                #if(set(path_keep[l]).intersection(set(path_keep[i]))):
                    #overlap_ind = np.asarray(np.sum(np.isin(np.asarray(path_keep[l]),np.asarray(path_keep[i])),axis=1)==2).nonzero()[0]
   
                    overlap_val = ([x for x in path_keep[l] if x in path_keep[i]])
  
                    overlap.extend(overlap_val)
            if(len(overlap)>0):        
                index_o= np.zeros(len(path_keep[l]),dtype=bool)
                for m in range(len(overlap)):
                    curI = np.where(np.sum(np.isin(np.asarray(path_keep[l]),np.asarray(overlap[m])),axis=1)==2)[0]
                    index_o[curI]=1
                    
                #index_o = np.sum(np.isin(np.asarray(path_keep[l]),np.asarray(overlap)),axis=1)==2
                
                overlap = [val for is_good, val in zip(index_o, path_keep[l]) if is_good]
                edges_to_remove = [val for is_good, val in zip(~index_o, path_keep[l]) if is_good]
    
                g_sub = graphTag.edge_subgraph(overlap).copy()
                for z in range(len(overlap)):
                    g_sub[overlap[z][0]][overlap[z][1]]['filament'] = tag
                    
                to_add.extend(list(g_sub.edges(data=True)))
            #remvove and tag edges
            if(len(edges_to_remove)>0):
                for m in range(len(edges_to_remove)):
                    G.remove_edge(edges_to_remove[m][0],edges_to_remove[m][1])
                    graphTag[edges_to_remove[m][0]][edges_to_remove[m][1]]['filament'] = tag
                # tag dangling edges
                graphTag[path_keep[l][0][0]][path_keep[l][0][1]]['filament dangling'] = 1
                graphTag[path_keep[l][-1][0]][path_keep[l][-1][1]]['filament dangling'] = 1
                
                tag += 1
                    
    # when all is done, create multi graph and add extra edges
    graphTagF = nx.MultiGraph(graphTag.copy())
    graphTagF.add_edges_from(to_add)
  
    return graphTagF

###############################################################################
#
# tracking - cost function
#
###############################################################################

def tagsU(graph):
    '''
    create list with unique filament tags

    Parameters
    ----------
    graph : nx graph
        graph with defined filament tags.

    Returns
    -------
    un_tags : list
        unique filament tags.

    '''
    film = list(graph.edges(data='filament'))
    filament_tag = [film[i][2] for i in range(len(film))]
    un_tags = np.unique(filament_tag)
    return un_tags

def pos_filament(graph,pos):
    '''
    for each unique filament tag, find the start and end nodes 

    Parameters
    ----------
    graph : nx graph
        graph with defined filament tags.
    pos : list
        list of node positions.

    Returns
    -------
    posF : list
        list of start and end nodes for each unique filament tag.

    '''
    
    unT = tagsU(graph)
    #cost matrix
    posF = np.zeros((len(unT),2))
    #find dangling nodes
    for l in range(len(unT)):
        #find start/end nodes in filaments
        nodesF = np.asarray([(a,b) for a,b, attrs in graph.edges(data=True) if (attrs["filament"] == l)]).flatten()
        countN = np.asarray(list(Counter(nodesF).items()))
        nodeES = [val for is_good, val in zip(countN[:,1]==1, countN[:,0]) if is_good]
    
        
        if(len(nodeES)==0):
            nodesF = np.asarray([(a,b) for a,b, attrs in graph.edges(data=True) if ((attrs["filament dangling"] == 1) and (attrs["filament"] == l))]).flatten()
            countN = np.asarray(list(Counter(nodesF).items()))
            nodeES = [val for is_good, val in zip(countN[:,1]==2, countN[:,0]) if is_good]
            if(len(nodeES)==0):
                #circle, and is emtpy
                nodeES = [nodesF[0]]
            nodeS1 = nodeES[0]
            nodeE1 = nodeES[0]
        elif(len(nodeES)==1):
            nodeS1 = nodeES[0]
            nodeE1 = nodeES[0]
        else:
            nodeS1 = nodeES[0]
            nodeE1 = nodeES[1]
            
        posF[l] = nodeS1,nodeE1
    return posF
      
def ass_cost(graph1,graph2,p1,p2):
    '''
    calculation for signmem function of cost

    Parameters
    ----------
    graph1 : nx graph
        graph from frame n.
    graph2 : nx graph
        graph from frame n+1.
    p1 : list
        list of node positions for fram n.
    p2 : list
        list of node positions for frame n+1.

    Returns
    -------
    value : float
        value for summed cost between two graphs.

    '''

    g1 = graph1.copy()
    g2 = graph2.copy()
            
    #create cost matrix
    posF1 =  pos_filament(g1,p1)
    posF2 = pos_filament(g2,p2)
    costM = np.zeros((max(len(posF1),len(posF2)),max(len(posF1),len(posF2))))
    for m in range(len(posF1)):
        nodeS1,nodeE1 = int(posF1[m][0]),int(posF1[m][1])
        
        for k in range(len(posF2)):
            nodeS2,nodeE2 = int(posF2[k][0]),int(posF2[k][1])
        
            dist1 = np.linalg.norm(p1[nodeS1]-p2[nodeS2])
            dist2 = np.linalg.norm(p1[nodeE1]-p2[nodeE2])
            dist_val12 = np.abs(dist1)+np.abs(dist2)
            #need to calculate the other way around too to check
            dist11 = np.linalg.norm(p1[nodeS1]-p2[nodeE2])
            dist22 = np.linalg.norm(p1[nodeE1]-p2[nodeS2])
            dist_val1122 = np.abs(dist11)+np.abs(dist22)
            #keep only the smallest one
            dist_val = min(dist_val12,dist_val1122)
            

            costM[m,k] += dist_val
        
    row_ind, col_ind = linear_sum_assignment(costM)
    value = costM[row_ind, col_ind].sum()

    return value

def signMem(graphWT,pos):
    '''
    Calculate the "memory" for keeping non connected filaments

    Parameters
    ----------
    graphWT : nx graph
        list of graphs .
    pos : list
        list of node positions for input list of graphs.

    Returns
    -------
    memKeep : int
        returns memory to keep filaments.

    '''
    # calculate memory of how many frames to use in assignment
    list_deltaT = []
    first=1
    memKeep=0
    #calculate the cost between frames
    for m in range(1, len(graphWT)):
        val=[]
        for i in range(len(graphWT)-1):
            if(i+m)<len(graphWT):    
                val.append((ass_cost(graphWT[i],graphWT[i+m],pos[i],pos[i+m]),m))
            
        list_deltaT.append(val)
    deltaT=np.asarray(list_deltaT[0])
    for m in range(1,len(deltaT)):
        curr = np.asarray(list_deltaT[m])
        deltaT = np.append(deltaT,curr,0)
    df_deltaT = pd.DataFrame(deltaT,columns=['deltaT','dist'])
    
    for i in range(2,len(graphWT)):
        _,p =stats.mannwhitneyu(df_deltaT[df_deltaT['dist']==1]['deltaT'], df_deltaT[df_deltaT['dist']==i]['deltaT'], method="auto")
        if((p<0.05) & (first==1)):
            first=0
            memKeep = i
            break
    return memKeep

def filament_tag(graph1,graph2,p1,p2,max_tag,max_cost,filamentsNU,memKeep):
    '''
    function for tracking filaments over frames, inputs the graphs of traced filaments, and gives the tracked graph back

    Parameters
    ----------
    graph1 : nx graph
        graph from frame n.
    graph2 : nx graph
        graph from frame n+1.
    p1 : list
        list of node positions for frame n.
    p2 : list
        list of node positions for frame n+1.
    max_tag : int
        max tagged value currently.
    max_cost : int
        Threshold value for max cost alloved for filaments.
    filamentsNU : list of filaments extra not tagged yet
        list of filaments extra not tagged yet.
    memKeep : int
        value, either set by user or by function.

    Returns
    -------
    g2 : nx graph
        DESCRIPTION.
    costM : matrix
        cost matrix.
    tag_new : int
        new max tag.
    filamentsNU : list
        list of filaments extra not tagged yet.

    '''
    
    save_index=[]
    for i in range(len(filamentsNU)):
        
        if(filamentsNU[i][3]>=memKeep):
            save_index.append(i)
        
        filamentsNU[i]= filamentsNU[i][0], filamentsNU[i][1],filamentsNU[i][2], filamentsNU[i][3]+1
    filamentsNU = [i for j, i in enumerate(filamentsNU) if j not in save_index]

    

    tag_new = max_tag
    g1 = graph1.copy()
    g2 = graph2.copy()
    unT1 = tagsU(g1)
    unT2 = tagsU(g2)
            
    #create cost matrix
    posF1 =  pos_filament(g1,p1)
    posF2 = pos_filament(g2,p2)
    costM = np.zeros((max(len(posF1),len(posF2)),max(len(posF1),len(posF2))))
    for m in range(len(posF1)):
        nodeS1,nodeE1 = int(posF1[m][0]),int(posF1[m][1])
        
        for k in range(len(posF2)):
            nodeS2,nodeE2 = int(posF2[k][0]),int(posF2[k][1])
        
            dist1 = np.linalg.norm(p1[nodeS1]-p2[nodeS2])
            dist2 = np.linalg.norm(p1[nodeE1]-p2[nodeE2])
            dist_val12 = np.abs(dist1)+np.abs(dist2)
            #need to calculate the other way around too to check
            dist11 = np.linalg.norm(p1[nodeS1]-p2[nodeE2])
            dist22 = np.linalg.norm(p1[nodeE1]-p2[nodeS2])
            dist_val1122 = np.abs(dist11)+np.abs(dist22)
            #keep only the smallest one
            dist_val = min(dist_val12,dist_val1122)
            

            costM[m,k] += dist_val
        
    row_ind, col_ind = linear_sum_assignment(costM)
    
    # If graph1 has more filaments than graph2
    if(len(posF1)>len(posF2)):
        for i in range(len(unT1)):
            #find the two filaments that are marked as the same from the linear assignment, and then mark no 2 with the value from no 1
            if(costM[row_ind[i],col_ind[i]] <= max_cost):
                tag_val = [(attrs['tags']) for a,b, attrs in g1.edges(data=True) if attrs["filament"] == row_ind[i]][0]
            else:
                check=0
                #check if the filament is close to one that could be saved.
                if(len(filamentsNU)>0):
                    dist_val=np.zeros(len(filamentsNU))
                    nodeS2,nodeE2 = int(posF2[col_ind[row_ind[i]]][0]),int(posF2[col_ind[row_ind[i]]][1])
                    for k in range(len(filamentsNU)):
                        
                        dist1 = np.linalg.norm(filamentsNU[k][0]-p2[nodeS2])
                        dist2 = np.linalg.norm(filamentsNU[k][1]-p2[nodeE2])
                        dist_val12 = np.abs(dist1)+np.abs(dist2)
                        #need to calculate the other way around too to check
                        dist11 = np.linalg.norm(filamentsNU[k][0]-p2[nodeE2])
                        dist22 = np.linalg.norm(filamentsNU[k][1]-p2[nodeS2])
                        dist_val1122 = np.abs(dist11)+np.abs(dist22)
                        #keep only the smallest one
                        dist_val[k] = min(dist_val12,dist_val1122)
                    ind = np.argmin(dist_val)
                    if(dist_val[ind]<=max_cost):
                        tag_val = filamentsNU[ind][2]  
                        #delete this position from filametnsNU
                        filamentsNU.pop(ind)
                        check=1
                if(check==0):
                    tag_val = tag_new+1
                    tag_new +=1
                    
                    nodeS,nodeE = int(posF1[row_ind[i]][0]),int(posF1[row_ind[i]][1])
                    
                    filamentsNU.append(((p1[nodeS]),p1[nodeE], [(attrs['tags']) for a,b, attrs in g1.edges(data=True) if attrs["filament"] == row_ind[i]][0],0))
                    
                
            fil_list2 = [(a,b) for a,b, attrs in g2.edges(data=True) if attrs["filament"] == col_ind[i]]
            
            
            for m in range(len(fil_list2)):
                once = True
                for n in range(len(g2[fil_list2[m][0]][fil_list2[m][1]])):
                    if(('tags' not in list(g2[fil_list2[m][0]][fil_list2[m][1]][n])) and (once==True)):
                        g2[fil_list2[m][0]][fil_list2[m][1]][n]['tags'] = tag_val
                        once = False
            if(len(fil_list2)==0):
                # the list of rows is larger than filaments in graph2, so we need to save the extra filament tags from graph1
                nodeS,nodeE = int(posF1[row_ind[i]][0]),int(posF1[row_ind[i]][1])
                filamentsNU.append(((p1[nodeS]),p1[nodeE], [(attrs['tags']) for a,b, attrs in g1.edges(data=True) if attrs["filament"] == row_ind[i]][0],0))
    # if graph2 has more filaments than graph1       
    else:

        for i in range(len(unT1)):
            if(costM[row_ind[i],col_ind[i]] <= max_cost):
                tag_val = [(attrs['tags']) for a,b, attrs in g1.edges(data=True) if attrs["filament"] == row_ind[i]][0]
            else:
                check=0
                #check if the filament is close to one that could be saved.
                if(len(filamentsNU)>0):
                    dist_val=np.zeros(len(filamentsNU))
                    nodeS2,nodeE2 = int(posF2[col_ind[row_ind[i]]][0]),int(posF2[col_ind[row_ind[i]]][1])
                    for k in range(len(filamentsNU)):
                        
                        dist1 = np.linalg.norm(filamentsNU[k][0]-p2[nodeS2])
                        dist2 = np.linalg.norm(filamentsNU[k][1]-p2[nodeE2])
                        dist_val12 = np.abs(dist1)+np.abs(dist2)
                        #need to calculate the other way around too to check
                        dist11 = np.linalg.norm(filamentsNU[k][0]-p2[nodeE2])
                        dist22 = np.linalg.norm(filamentsNU[k][1]-p2[nodeS2])
                        dist_val1122 = np.abs(dist11)+np.abs(dist22)
                        #keep only the smallest one
                        dist_val[k] = min(dist_val12,dist_val1122)
                    ind = np.argmin(dist_val)
                    if(dist_val[ind]<=max_cost):
                        tag_val = filamentsNU[ind][2]  
                        filamentsNU.pop(ind)
                        check=1
                if(check==0):
                    tag_val = tag_new+1
                    tag_new +=1
                    
                    nodeS,nodeE = int(posF1[row_ind[i]][0]),int(posF1[row_ind[i]][1])
                    filamentsNU.append(((p1[nodeS]),p1[nodeE], [(attrs['tags']) for a,b, attrs in g1.edges(data=True) if attrs["filament"] == row_ind[i]][0],0))
    
            
            #fil_list1 = [(a,b) for a,b, attrs in g1.edges(data=True) if attrs["filament"] == row_ind[i]]
            fil_list2 = [(a,b) for a,b, attrs in g2.edges(data=True) if attrs["filament"] == col_ind[i]]
            
            for m in range(len(fil_list2)):
                once = True
                for n in range(len(g2[fil_list2[m][0]][fil_list2[m][1]])):
                    if(('tags' not in list(g2[fil_list2[m][0]][fil_list2[m][1]][n])) and (once==True)):
                        g2[fil_list2[m][0]][fil_list2[m][1]][n]['tags'] = tag_val
                        once = False
                        
        for s in range(i+1,len(unT2)):
            #check if filamentsNU has filament tags
            check=0
            if(len(filamentsNU)>0):
                dist_val=np.zeros(len(filamentsNU))
                nodeS2,nodeE2 = int(posF2[col_ind[row_ind[s]]][0]),int(posF2[col_ind[row_ind[s]]][1])
                for k in range(len(filamentsNU)):
                    
                    dist1 = np.linalg.norm(filamentsNU[k][0]-p2[nodeS2])
                    dist2 = np.linalg.norm(filamentsNU[k][1]-p2[nodeE2])
                    dist_val12 = np.abs(dist1)+np.abs(dist2)
                    #need to calculate the other way around too to check
                    dist11 = np.linalg.norm(filamentsNU[k][0]-p2[nodeE2])
                    dist22 = np.linalg.norm(filamentsNU[k][1]-p2[nodeS2])
                    dist_val1122 = np.abs(dist11)+np.abs(dist22)
                    #keep only the smallest one
                    dist_val[k] = min(dist_val12,dist_val1122)
                ind = np.argmin(dist_val)
                if(dist_val[ind]<=max_cost):
                    tag_val = filamentsNU[ind][2]  
                    filamentsNU.pop(ind)
                    check=1
            if(check==0):
                tag_val = tag_new+1
                tag_new +=1
            
            #print(k)
            fil_list2 = [(a,b) for a,b, attrs in g2.edges(data=True) if attrs["filament"] == col_ind[s]]
            for m in range(len(fil_list2)):
                #g2[fil_list2[m][0]][fil_list2[m][1]]['tags'] = tag_new+1
                once = True
                for n in range(len(g2[fil_list2[m][0]][fil_list2[m][1]])):
                    if(('tags' not in list(g2[fil_list2[m][0]][fil_list2[m][1]][n])) and (once==True)):
                        g2[fil_list2[m][0]][fil_list2[m][1]][n]['tags'] = tag_val
                        once = False


    return g2, costM, tag_new, filamentsNU
 
###############################################################################
#
# drawing functions
#
###############################################################################

def draw_graph_filament_nocolor(image,graph,pos,title,value):
    '''
    drawing function

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    saved image

    '''
    edges,values = zip(*nx.get_edge_attributes(graph,value).items())
    plt.figure(figsize=(8,10))
    plt.title(title)
    plt.imshow(image,cmap='gray')
    vmin=min(values)
    vmax=max(values)+1
    if(vmax%2==0):
        vmax +=1
    cmap=plt.get_cmap('tab20',vmax)
    nx.draw(graph,pos, edge_cmap=cmap, edge_color=values,node_size=0.7,width=3,alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    #plt.colorbar(sm,orientation='horizontal')
    plt.tight_layout()
    return


def draw_graph_filament_nocolor_r(image,graph,pos,title,value):
    '''
    drawing function

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    saved image

    '''
    edges,values = zip(*nx.get_edge_attributes(graph,value).items())
    plt.figure(figsize=(8,10))
    plt.title(title)
    plt.imshow(image,cmap='gray_r')
    vmin=min(values)
    vmax=max(values)+1
    if(vmax%2==0):
        vmax +=1
    cmap=plt.get_cmap('tab20',vmax)
    nx.draw(graph,pos, edge_cmap=cmap, edge_color=values,node_size=0.7,width=3,alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    #plt.colorbar(sm,orientation='horizontal')
    plt.tight_layout()
    return

def draw_graph_filament_nocolor_size(image,graph,pos,title,value,size):
    '''
    drawing function with size of window

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    saved image.

    '''
    edges,values = zip(*nx.get_edge_attributes(graph,value).items())
    plt.figure(figsize=size)
    plt.title(title)
    plt.imshow(image,cmap='gray')
    vmin=min(values)
    vmax=max(values)+1
    if(vmax%2==0):
        vmax +=1
    cmap=plt.get_cmap('tab20',vmax)
    nx.draw(graph,pos, edge_cmap=cmap, edge_color=values,node_size=0.7,width=3,alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    #plt.colorbar(sm,orientation='horizontal')
    plt.tight_layout()
    return

def draw_graph(image,graph,pos,title):
    '''
    simple drawing function

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(image,cmap='gray')
    #plt.colorbar()
    nx.draw(graph,pos,cmap=plt.get_cmap('viridis'),node_size=30,with_labels=True,edge_color='red', font_color='white',font_size=14,alpha=0.5)
    #plt.legend()
    return

def draw_graph_filament_track_nocolor(image,graph,pos,title,max_tag,padv):
    '''
    drawing function for tagged filaments, keep the color of filaments the same over frames

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    graph : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    max_tag : TYPE
        DESCRIPTION.
    padv : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    edges,values = zip(*nx.get_edge_attributes(graph,'tags').items())
    fig, axs = plt.subplots(1, 1, constrained_layout=True,figsize=(10,10))
    #plt.title(title)
    imageP = np.pad(image, padv, 'constant')
    posP=pos+padv
    plt.imshow(imageP,cmap='gray')
    vmax=max_tag
    
    cmapV=plt.get_cmap('tab20',vmax+1)
    
    cNorm  = colors.Normalize(vmin=0, vmax=max_tag)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmapV)
    colorList = []
 
    for i in range(len(values)):
       colorVal = scalarMap.to_rgba(values[i])
       colorList.append(colorVal)
    
    options = {
        
        "node_size": 0.7,
        "edge_color": colorList,
        #"edge_cmap": cmap,
        "width": 3,
        'alpha': 0.8
        }
    
    nx.draw(graph,posP, **options)

    return

 
###############################################################################
#
# post processing
#
###############################################################################

def mask2rot(mask):
    '''
    calculation of major cell axis

    Parameters
    ----------
    mask : array
        mask of cell binary image.

    Returns
    -------
    directionVector : float
        angle for major cell axis.

    '''

    skeletonizedMask = skimage.morphology.skeletonize(mask)
    coordinatesSkeleton = np.array(np.where(skeletonizedMask > 0)).T[:, ::-1]
    pointsOnSkeleton = int(len(coordinatesSkeleton) * 0.2)
    coordinateCellAxis1 = coordinatesSkeleton[pointsOnSkeleton]
    coordinateCellAxis2 = coordinatesSkeleton[-pointsOnSkeleton]
    directionVector = coordinateCellAxis2 - coordinateCellAxis1
    return directionVector

def angle_majorCell(nodes_edge,nodePositions,vec_mask):
    '''
    angle of filament from major cell axis

    Parameters
    ----------
    nodes_edge : list
        DESCRIPTION.
    nodePositions : list
        positions of nodes.
    vec_mask : float
        angle for major cell axis.

    Returns
    -------
    angle : float
        angle for filament from major cell axis.

    '''
    
    vec1 = nodePositions[int(nodes_edge[1])] - nodePositions[int(nodes_edge[0])]
    vec2 = vec_mask
    if((vec1==0).all()):
        angle=0
    else:
        angle=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) * 180.0 / np.pi
    return angle

def barplot180(list_points, list_bins, output_path, color_code):
    '''
    creation of circular barplot for angles

    Parameters
    ----------
    list_points : list
        list of binned weighted angles.
    list_bins : list
        list of bin values.
    output_path : str
        path to save figure.
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
    
    fig.write_image(output_path, format='png')
    return

def filament_info_time(img_o, g_tagged, posL, path,imF,maskDraw):
    '''
    creation of dataframe to be saved as csv of all filament info of tracked and traced filaments

    Parameters
    ----------
    img_o : array
        array of raw image timeseries.
    g_tagged : nx graph
        list of tracked grapha.
    posL : list
        list of list of node positions.
    path : directory path
        path to save file.
    imF : arrray
        skeletonized image.
    maskDraw : array
        binary mask of cell.

    Returns
    -------
    FullfilInfo : dataframe
        dataframe containing all the timeseries information.

    '''
    
    fil_len = [0]*len(img_o)
    fil_lenvar = [0]*len(img_o)
    fil_density = [0]*len(img_o)
    fil_densityvar = [0]*len(img_o)
    fil_rod = [0]*len(img_o)
    fil_rodvar = [0]*len(img_o)
    fil_bend = [0]*len(img_o)
    fil_bendvar = [0]*len(img_o)
    
    FullfilInfo = pd.DataFrame()
    
    for m in range(len(img_o)):
        frame = m
        filamentTags = np.unique(np.asarray(list(g_tagged[m].edges(data='tags')))[:,2])
        filInfo = pd.DataFrame()
        filamentT = nx.to_pandas_edgelist(g_tagged[m])
        nodesSE = pos_filament(g_tagged[m],posL[m])
        fullL = np.zeros(len(filamentTags))
        nodeD = np.zeros(len(filamentTags))
        fullI = np.zeros(len(filamentTags))
        fullC = np.zeros(len(filamentTags))
        fullBL = np.zeros(len(filamentTags))
        fullE = np.zeros(len(filamentTags))
        fullEd = np.zeros(len(filamentTags))
        fullDe = np.ones(len(filamentTags))
        tags = np.ones(len(filamentTags))
        vec_mask = mask2rot(maskDraw)
        #in case its a perfect square
        if(all(vec_mask==[0,0])):
            vec_mask=[1,0]
        for i,l in zip(filamentTags,range(len(filamentTags))):
            #filament length
            fullL[l] = np.sum(filamentT['fdist'][filamentT['tags']==i])
            fullE[l] = np.sum(filamentT['edist'][filamentT['tags']==i])
            #node distance
            nodeD[l] = np.linalg.norm(posL[m][int(nodesSE[l][0])]-posL[m][int(nodesSE[l][1])])
            #filament intensity
            fullI[l] = np.sum(filamentT['weight'][filamentT['tags']==i])
            #filament intensity by filament length
            fullC[l] = fullI[l]/fullL[l]
            #rod length over filament length
            fullBL[l] = nodeD[l]/fullL[l]
            if(fullBL[l]>1):
                fullBL[l]=1
            #angle from major cell axis
            edge_ang = angle_majorCell(nodesSE[l],posL[m],vec_mask)
            tags[l] = i 
            fullEd[l] = edge_ang #min(edge_ang,np.abs(edge_ang-180))
            # density:
        fullDe = fullDe*np.sum(imF[m]*1)/(np.sum(maskDraw))
            
        
        filInfo['filament'] = tags
        filInfo['filament length'] = fullL
        filInfo['filament edist'] = fullE
        filInfo['filament rod length'] = nodeD
        filInfo['filament intensity'] = fullI
        filInfo['filament intensity per length'] = fullC
        filInfo['filament bendiness'] = fullBL
        filInfo['filament angle'] = fullEd
        filInfo['frame number'] = frame
        filInfo['filament density'] = fullDe
        
        FullfilInfo = pd.concat(([FullfilInfo,filInfo]))
        
        
        
        fil_len[m] =np.median(filInfo['filament length'])
        fil_lenvar[m] = stats.median_abs_deviation(filInfo['filament length'])
        
        fil_density[m] = np.mean(filInfo['filament intensity per length'])
        fil_densityvar[m] = np.std(filInfo['filament intensity per length'])
        
        fil_rod[m] = np.median(filInfo['filament rod length'])
        fil_rodvar[m] = stats.median_abs_deviation(filInfo['filament rod length'])
        
        fil_bend[m] = np.mean(filInfo['filament bendiness'])
        fil_bendvar[m] = np.std(filInfo['filament bendiness'])
    
    FullfilInfo.to_csv(os.path.join(path, 'tracked_filaments_info.csv'),index=False)
    
    return FullfilInfo

def filament_info(img_o, graphTagg, posL, path,imF,maskDraw):
    '''
    creation of dataframe to be saved as csv of all filament info of traced filaments

    Parameters
    ----------
    img_o : array
        array of raw image timeseries.
    graphTagg : nx graph
        list of traced graph.
    posL : list
        list of list of node positions.
    path : directory path
        path to save file.
    imF : arrray
        skeletonized image.
    maskDraw : array
        binary mask of cell.

    Returns
    -------
    FullfilInfo : dataframe
        dataframe containing all the data-image information.
    '''

    FullfilInfo = pd.DataFrame()


    filamentTags = np.unique(np.asarray(list(graphTagg.edges(data='filament')))[:,2])
    filInfo = pd.DataFrame()
    filamentT = nx.to_pandas_edgelist(graphTagg)
    nodesSE = pos_filament(graphTagg,posL)
    fullL = np.zeros(len(filamentTags))
    nodeD = np.zeros(len(filamentTags))
    fullI = np.zeros(len(filamentTags))
    fullC = np.zeros(len(filamentTags))
    fullBL = np.zeros(len(filamentTags))
    fullE = np.zeros(len(filamentTags))
    fullEd = np.zeros(len(filamentTags))
    fullDe = np.ones(len(filamentTags))
    tags = np.ones(len(filamentTags))
    vec_mask = mask2rot(maskDraw)
    #in case its a perfect square
    if(all(vec_mask==[0,0])):
        vec_mask=[1,0]
    for i,l in zip(filamentTags,range(len(filamentTags))):
        #filament length
        fullL[l] = np.sum(filamentT['fdist'][filamentT['filament']==i])
        fullE[l] = np.sum(filamentT['edist'][filamentT['filament']==i])
        #node distance
        nodeD[l] = np.linalg.norm(posL[int(nodesSE[l][0])]-posL[int(nodesSE[l][1])])
        #filament intensity
        fullI[l] = np.sum(filamentT['weight'][filamentT['filament']==i])
        #filament intensity by filament length
        fullC[l] = fullI[l]/fullL[l]
        #rod length over filament length
        fullBL[l] = nodeD[l]/fullL[l]
        #angle from major cell axis
        edge_ang = angle_majorCell(nodesSE[l],posL,vec_mask)
        tags[l] = i 
        fullEd[l] = edge_ang #min(edge_ang,np.abs(edge_ang-180))
        # density:
    fullDe = fullDe*np.sum(imF*1)/(np.sum(maskDraw))
        
    
    filInfo['filament'] = tags
    filInfo['filament length'] = fullL
    filInfo['filament edist'] = fullE
    filInfo['filament rod length'] = nodeD
    filInfo['filament intensity'] = fullI
    filInfo['filament intensity per length'] = fullC
    filInfo['filament bendiness'] = fullBL
    filInfo['filament angle'] = fullEd
    filInfo['filament density'] = fullDe
    
    FullfilInfo = filInfo
        
    FullfilInfo.to_csv(os.path.join(path, 'traced_filaments_info.csv'),index=False)
    
    return FullfilInfo


def filament_info_no(img_o, graphTagg, posL, path,imF,maskDraw,no):
    '''
    creation of dataframe to be saved as csv of all filament info of traced filaments

    Parameters
    ----------
    img_o : array
        array of raw image timeseries.
    graphTagg : nx graph
        list of traced graph.
    posL : list
        list of list of node positions.
    path : directory path
        path to save file.
    imF : arrray
        skeletonized image.
    maskDraw : array
        binary mask of cell.

    Returns
    -------
    FullfilInfo : dataframe
        dataframe containing all the data-image information.
    '''

    FullfilInfo = pd.DataFrame()


    filamentTags = np.unique(np.asarray(list(graphTagg.edges(data='filament')))[:,2])
    filInfo = pd.DataFrame()
    filamentT = nx.to_pandas_edgelist(graphTagg)
    nodesSE = pos_filament(graphTagg,posL)
    fullL = np.zeros(len(filamentTags))
    nodeD = np.zeros(len(filamentTags))
    fullI = np.zeros(len(filamentTags))
    fullC = np.zeros(len(filamentTags))
    fullBL = np.zeros(len(filamentTags))
    fullE = np.zeros(len(filamentTags))
    fullEd = np.zeros(len(filamentTags))
    fullDe = np.ones(len(filamentTags))
    tags = np.ones(len(filamentTags))
    vec_mask = mask2rot(maskDraw)
    #in case its a perfect square
    if(all(vec_mask==[0,0])):
        vec_mask=[1,0]
    for i,l in zip(filamentTags,range(len(filamentTags))):
        #filament length
        fullL[l] = np.sum(filamentT['fdist'][filamentT['filament']==i])
        fullE[l] = np.sum(filamentT['edist'][filamentT['filament']==i])
        #node distance
        nodeD[l] = np.linalg.norm(posL[int(nodesSE[l][0])]-posL[int(nodesSE[l][1])])
        #filament intensity
        fullI[l] = np.sum(filamentT['weight'][filamentT['filament']==i])
        #filament intensity by filament length
        fullC[l] = fullI[l]/fullL[l]
        #rod length over filament length
        fullBL[l] = nodeD[l]/fullL[l]
        #angle from major cell axis
        edge_ang = angle_majorCell(nodesSE[l],posL,vec_mask)
        tags[l] = i 
        fullEd[l] = edge_ang #min(edge_ang,np.abs(edge_ang-180))
        # density:
    fullDe = fullDe*np.sum(imF*1)/(np.sum(maskDraw))
        
    
    filInfo['filament'] = tags
    filInfo['filament length'] = fullL
    filInfo['filament edist'] = fullE
    filInfo['filament rod length'] = nodeD
    filInfo['filament intensity'] = fullI
    filInfo['filament intensity per length'] = fullC
    filInfo['filament bendiness'] = fullBL
    filInfo['filament angle'] = fullEd
    filInfo['filament density'] = fullDe
    
    FullfilInfo = filInfo
        
    FullfilInfo.to_csv(os.path.join(path, no+'traced_filaments_info.csv'),index=False)
    
    return FullfilInfo

def circ_stat_plot(path,pd_fil_info):
    '''
    calculate circular statitstics and return mean circular angle and variance and save all figures

    Parameters
    ----------
    path : directory path
        path to save figure.
    pd_fil_info : dataframe
        dataframe of information of tracked filaments.

    Returns
    -------
    mean_angle : float
        circular mean angle.
    var_val : float
        cirvular variance for mean angle.

    '''
    
    frame_no = pd_fil_info['frame number'].unique()
    mean_angle = np.zeros(len(frame_no))
    var_val = np.zeros(len(frame_no))
    for m in range(len(frame_no)):
        
        data = np.asarray(pd_fil_info[pd_fil_info['frame number']==m]['filament angle'])*u.deg
        weight = pd_fil_info[pd_fil_info['frame number']==m]['filament length']
        mean_angle[m] = np.asarray((astropy.stats.circmean(data,weights=weight)))
        var_val[m] = np.asarray(astropy.stats.circvar(data,weights=weight))
        
        hist180,bins180 = np.histogram(0,int(180/5),[0,180])
        
        list_ec = np.zeros(len(bins180[1:]))
        for l in range(len(bins180[1:])):
        
            list_ec[l] = pd_fil_info[pd_fil_info['frame number']==m]['filament length'][(pd_fil_info[pd_fil_info['frame number']==m]['filament angle']>bins180[l]) & (pd_fil_info[pd_fil_info['frame number']==m]['filament angle']<=bins180[l+1])].sum()
        
        bins180 = bins180[1:]-2.5
        path180 = os.path.join(path, "circ_stat", f"stat{m}.png")
        barplot180(list_ec, bins180, path180, color_code='#0000FF')

    return mean_angle,var_val

def circ_stat_time(pd_fil_info):
    '''
    alculate circular statitstics and return mean circular angle and variance

    Parameters
    ----------
    pd_fil_info : dataframe
        dataframe of information of tracked filaments.

    Returns
    -------
    mean_angle : float
        circular mean angle.
    var_val : float
        cirvular variance for mean angle.

    '''
    
    frame_no = pd_fil_info['frame number'].unique()
    mean_angle = np.zeros(len(frame_no))
    var_val = np.zeros(len(frame_no))
    for m in range(len(frame_no)):
        
        data = np.asarray(pd_fil_info[pd_fil_info['frame number']==m]['filament angle'])*u.deg
        weight = pd_fil_info[pd_fil_info['frame number']==m]['filament length']
        mean_angle[m] = np.asarray((astropy.stats.circmean(data,weights=weight)))
        var_val[m] = np.asarray(astropy.stats.circvar(data,weights=weight))
        
    return mean_angle,var_val

def circ_stat(pd_fil_info,path):

    data = np.asarray(pd_fil_info['filament angle'])*u.deg
    weight = pd_fil_info['filament length']
    mean_angle = np.asarray((astropy.stats.circmean(data,weights=weight)))
    var_val = np.asarray(astropy.stats.circvar(data,weights=weight))
        
    hist180,bins180 = np.histogram(0,int(180/5),[0,180])
        
    list_ec = np.zeros(len(bins180[1:]))
    for l in range(len(bins180[1:])):
    
        list_ec[l] = pd_fil_info['filament length'][(pd_fil_info['filament angle']>bins180[l]) & (pd_fil_info['filament angle']<=bins180[l+1])].sum()
    
     
    bins180 = bins180[1:]-2.5
    path180 = os.path.join(path, "circ_stat", "stat.png")
    barplot180(list_ec, bins180, path180, color_code='#0000FF')
    return mean_angle,var_val

def pos_tagged_filament(graph,pos,l):

    #find start/end nodes in filaments
    nodesF = np.asarray([(a,b) for a,b, attrs in graph.edges(data=True) if (attrs["tags"] == l)]).flatten()
    countN = np.asarray(list(Counter(nodesF).items()))
    nodeES = [val for is_good, val in zip(countN[:,1]==1, countN[:,0]) if is_good]

    
    if(len(nodeES)==0):
        nodesF = np.asarray([(a,b) for a,b, attrs in graph.edges(data=True) if ((attrs["filament dangling"] == 1) and (attrs["tags"] == l))]).flatten()
        countN = np.asarray(list(Counter(nodesF).items()))
        nodeES = [val for is_good, val in zip(countN[:,1]==2, countN[:,0]) if is_good]
        if(len(nodeES)==0):
            #circle, and is emtpy
            nodeES = [nodesF[0]]
        nodeS1 = nodeES[0]
        nodeE1 = nodeES[0]
    elif(len(nodeES)==1):
        nodeS1 = nodeES[0]
        nodeE1 = nodeES[0]
    else:
        nodeS1 = nodeES[0]
        nodeE1 = nodeES[1]
        
    posF = nodeS1,nodeE1
    return posF

def calc_dist_end(SE_1,SE_2,posL,m1,m2):
    nodeS1,nodeE1 = int(SE_1[0]),int(SE_1[1])
    nodeS2,nodeE2 = int(SE_2[0]),int(SE_2[1])
    dist1 = np.linalg.norm(posL[m1][nodeS1]-posL[m2][nodeS2])
    dist11 = np.linalg.norm(posL[m1][nodeS1]-posL[m2][nodeE2])
    dist2 = np.linalg.norm(posL[m1][nodeE1]-posL[m2][nodeE2])
    dist22 = np.linalg.norm(posL[m1][nodeE1]-posL[m2][nodeS2])
    distS = dist11
    distE = dist22
    if(dist1<=dist11):
        distS=dist1
    if(dist2<=dist22):
        distE=dist2
    return distS,distE

def angle_move(fullTrack,pd_fil_info):
    pd_angle = pd.DataFrame()
    mAL_allA =[]
    mAL_allT =[]
    mAL_allF =[]
    for l in range(0,int(np.max(fullTrack['filament tag']))+1):
        angle_fil = pd_fil_info['filament angle'][pd_fil_info['filament']==l].values
        frame_fil = pd_fil_info['frame number'][pd_fil_info['filament']==l].values
        mAL=np.zeros((len(angle_fil)-1,3))
        for m in range(len(angle_fil)-1):
            mAL[m,0] = angle_fil[m]-angle_fil[m+1]
            mAL[m,1] = l
            mAL[m,2] = frame_fil[m+1]
        mAL_allA = np.append(mAL_allA,mAL[:,0])
        mAL_allT = np.append(mAL_allT,mAL[:,1])
        mAL_allF = np.append(mAL_allF,mAL[:,2])
        
    pd_angle['filament tag'] = mAL_allT
    pd_angle['change angle'] = mAL_allA
    pd_angle['frame'] = mAL_allF
            
    result = pd.merge(fullTrack, pd_angle,on=['filament tag','frame'])
    return result

def track_move(g_tagged,posL,img_o,memKeep, max_cost,path,pd_fil_info):
    fil_keep=[]
    fullTrack = pd.DataFrame()
    #do it for many frames here
    for y in range(len(img_o)-1):
        frameTrack = pd.DataFrame()
        #find unique tags for current and next frame
        tags = np.unique(np.asarray(list(g_tagged[y].edges(data='tags')))[:,2])
        tags2 = np.unique(np.asarray(list(g_tagged[y+1].edges(data='tags')))[:,2])
        
        tagsM = []
        trackMean = []
        trackMax = []
        if(len(fil_keep)>0):
            rem_index = []
            for n in range(len(fil_keep)):
                
                save_index=[]
                if(y-fil_keep[n][0]>=memKeep):
                    save_index.append(n)
                    #filamentsNU.pop(i)
                
                #fil_keep[n][0]+=1
            fil_keep = [i for j, i in enumerate(fil_keep) if j not in save_index]
            #run this for the kept filaments
            for s in range(len(fil_keep)):
                indexFil = fil_keep[s][1]
                if(indexFil in tags2):
                    
                    tagsM.append(indexFil)
                    distP=[]
                    SE_1 = fil_keep[s][2]
                    SE_2 = pos_tagged_filament(g_tagged[y+1],posL[y+1],indexFil)
                    distS,distE = calc_dist_end(SE_1,SE_2,posL,fil_keep[s][0],y+1)
                    distP.append(distS)
                    distP.append(distE)
                    #remove it again
                    rem_index.append(s)
                    
                    #nodes from the filament at time t
                    nodes = fil_keep[s][-1]
                    # edges for the filament matched in time t+1
                    edges2 = [(u,v) for u,v,e in g_tagged[y+1].edges(data=True) if e['tags'] == indexFil] 
                    
                    
                    for m in range(len(nodes)):
                        m_switch=0
                        on=0
                        for k in range(len(edges2)):
                            if(m_switch==0):
                        
                                point1 = Point(posL[fil_keep[s][0]][nodes[m]])
                                if((posL[fil_keep[s][0]][nodes[m]]==posL[y+1][edges2[k][0]]).all() or (posL[fil_keep[s][0]][nodes[m]]==posL[y+1][edges2[k][1]]).all()):
                                    #print('they are the same', m,k)
                                    m_switch=1
                                    on=0
                                else:
                                    line2 = Line.from_points(posL[y+1][edges2[k][0]],posL[y+1][edges2[k][1]])
                                    
                                    point_projected = line2.project_point(point1)
                                    if((point_projected!=point1).all()):
                                
                                        lineT = [posL[y+1][edges2[k][0]],posL[y+1][edges2[k][1]]]
                                        if((np.abs(lineT[1]-lineT[0]) == np.abs(lineT[1]-point_projected) + np.abs(lineT[0]-point_projected)).all()):
                                            #print("it is lying on the line!",m, k)
                                            #distP.append(np.linalg.norm(point1-point_projected))
                                            keepM = np.linalg.norm(point1-point_projected)
                                            if(keepM<max_cost):
                                                on=1
                        if((m_switch==0) and (on==1)):
                            distP.append(keepM)
    
                    trackMean.append(np.mean(distP))
                    trackMax.append(np.max(distP))
            #remove
            fil_keep = [i for j, i in enumerate(fil_keep) if j not in rem_index]
     
        for l in tags:
            if(l in tags2):
                tagsM.append(l)
                
                distP=[]
                SE_1 = pos_tagged_filament(g_tagged[y],posL[y],l)
                SE_2 = pos_tagged_filament(g_tagged[y+1],posL[y+1],l)
                distS,distE = calc_dist_end(SE_1,SE_2,posL,y,y+1)
                distP.append(distS)
                distP.append(distE)
                
                #nodes from the filament at time t
                edges = [(u,v) for u,v,e in g_tagged[y].edges(data=True) if e['tags'] == l] 
                nodes = np.unique(np.asarray(edges).flatten())
                # edges for the filament matched in time t+1
                edges2 = [(u,v) for u,v,e in g_tagged[y+1].edges(data=True) if e['tags'] == l] 
    
                for m in range(len(nodes)):
                    m_switch=0
                    on=0
                    for k in range(len(edges2)):
                        if(m_switch==0):
                    
                            point1 = Point(posL[y][nodes[m]])
                            if((posL[y][nodes[m]]==posL[y+1][edges2[k][0]]).all() or (posL[y][nodes[m]]==posL[y+1][edges2[k][1]]).all()):
                                #print('they are the same', m,k)
                                m_switch=1
                            else:
                                line2 = Line.from_points(posL[y+1][edges2[k][0]],posL[y+1][edges2[k][1]])
                                
                                point_projected = line2.project_point(point1)
                                if((point_projected!=point1).all()):
                            
                                    lineT = [posL[y+1][edges2[k][0]],posL[y+1][edges2[k][1]]]
                                    #check if the point is lying on the line created by the edge, if yes save
                                    if((np.abs(lineT[1]-lineT[0]) == np.abs(lineT[1]-point_projected) + np.abs(lineT[0]-point_projected)).all()):
                                        #must keep it in memory, in case two lines will be overlaying
                                        keepM = np.linalg.norm(point1-point_projected)
                                        #distP.append(np.linalg.norm(point1-point_projected))
                                        if(keepM<max_cost):
                                            on=1
                    if((m_switch==0) and (on==1)):
                        distP.append(keepM)
    
                trackMean.append(np.mean(distP))
                trackMax.append(np.max(distP))
                
            else:
                #we need to save this filament to check distance in next frame if exists.
                SE_1 = pos_tagged_filament(g_tagged[y],posL[y],l)
                edges = [(u,v) for u,v,e in g_tagged[y].edges(data=True) if e['tags'] == l] 
                nodes = np.unique(np.asarray(edges).flatten())
                fil_keep.append([y,l,SE_1,nodes])
                
        frameTrack['filament tag'] = tagsM
        frameTrack['mean move'] = trackMean
        frameTrack['max move'] = trackMax
        frameTrack['frame'] = y+1
        #concat the pandas together
        fullTrack = pd.concat(([fullTrack,frameTrack]),ignore_index=True)
    pd_full = angle_move(fullTrack,pd_fil_info)
    pd_full.to_csv(os.path.join(path, 'tracked_move.csv'),index=False)
    return pd_full
