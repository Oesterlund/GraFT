#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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

path="where_dataUtils_is"
sys.path.append(path)

import dataUtils

path2="pathtoutilsf"
sys.path.append(path2)

import utilsF

overpath = 'define_this_path'

plt.close('all')

box_size = 500

###############################################################################
#
# function
#
###############################################################################

def use_GraFT(imgs_list,path_imgs,box_size,graphSave,size,eps,thresh_top,sigma,small,listtP,tT2,s2):

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
            if(countGraph in listtP):
                graph_s[q], posL[q], imgSkel[q], imgAF[q], imgBl[q],imF[q],mask[q],df_pos[q] = utilsF.creategraph(img_o[q],size=size,eps=eps,thresh_top=tT2,sigma=sigma,small=s2)
            else:                
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
                
                dataUtils.draw_graph_filament_nocolor(imgSkel[q],graphTagg[q],posL[q],"",'filament')
                plt.savefig(path_imgs+'graph/n_graphs/graph{0}.png'.format(countGraph))
       
                #print('filament defined: ',len(np.unique(np.asarray(list(graphTagg[q].edges(data='filament')))[:,2])))
                
                correctCount[q],countIm[q],count[q],TPI[q],FPI[q],FNI[q] = dataUtils.calcDiff(img_linesPad[q],mask[q],df_pos[q])
            
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
        
        
        JI_list = dataUtils.JI(TPI,FPI,FNI)
        JI_all.append(JI_list)
        df_JI = pd.DataFrame({
            'frame': np.arange(len(img_o)),
            'JI': JI_list
        })
        df_JI.to_csv(path_imgs + 'JI/{0}_JI.csv'.format(zx), index=False) 

    return

###############################################################################
#
# use GraFT to create graphs
#
###############################################################################

# load in test data
graphSave = overpath+'/noise001/noise10/graph/'

path_imgs = overpath+'/noise001/noise10/'

imgs_list = ['100_lines_intermediate.tiff','200_lines_intermediate.tiff','300_lines_intermediate.tiff','400_lines_intermediate.tiff','500_lines_intermediate.tiff']

csv_list = ['100_lines_intermediate_line_position.csv','200_lines_intermediate_line_position.csv','300_lines_intermediate_line_position.csv',
              '400_lines_intermediate_line_position.csv','500_lines_intermediate_line_position.csv']

#create tiff files
#dataUtils.create_tiffimgs(imgs_list = imgs_list,path_imgs = path_imgs)
listCt = np.array([107, 156, 266])
#size=6,eps=100,thresh_top=0.4,sigma=1,small=50
use_GraFT(imgs_list=imgs_list,path_imgs=path_imgs,box_size=500,graphSave= graphSave
                    ,size=6,eps=100,thresh_top=0.35,sigma=1,small=30,listtP = listCt,tT2=0.9,s2 = 50)

###############################################################################
#
# identification for 10 lines data
#
###############################################################################

savepathLines = overpath+'/noise001/noise10/line_comparisons/'

files_csv = overpath+'/noise001/noise10/'

imgs_list = ['100_lines_intermediate.tiff','200_lines_intermediate.tiff','300_lines_intermediate.tiff','400_lines_intermediate.tiff','500_lines_intermediate.tiff']

csv_list = ['100_lines_intermediate_line_position.csv','200_lines_intermediate_line_position.csv','300_lines_intermediate_line_position.csv',
            '400_lines_intermediate_line_position.csv','500_lines_intermediate_line_position.csv']

graph_path = overpath+'/noise001/noise10/graph/'
graph_list = ['0_graphTagg.gpickle','1_graphTagg.gpickle','2_graphTagg.gpickle','3_graphTagg.gpickle','4_graphTagg.gpickle']

pos_list = ['0_posL.gpickle','1_posL.gpickle','2_posL.gpickle','3_posL.gpickle','4_posL.gpickle']

dataUtils.identification_LAP(csv_list=csv_list,files_csv=files_csv,savepathLines=savepathLines,graph_path=graph_path,graph_list=graph_list,pos_list=pos_list,box_size=500)
