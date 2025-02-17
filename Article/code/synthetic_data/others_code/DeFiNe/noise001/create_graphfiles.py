#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: isabella
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pickle

overpath = 'define_this_path'

path2=overpath+"/others_code/DeFiNe/"
sys.path.append(path2)
import create_gml


plt.close('all')

def create_graphgml(graph, path,pos):
    g3=graph.copy()
    simpleGraph = nx.Graph(g3)
    edges= list(simpleGraph.edges())
    tag=0
    for i in range(len(simpleGraph.edges())):
        simpleGraph[edges[i][0]][edges[i][1]]['manu'] = '{}-{}'.format(tag,0)
        simpleGraph[edges[i][0]][edges[i][1]]['auto'] = '{}-{}'.format(tag,0)
        tag+=1
      
    graphW = create_gml.strG(simpleGraph)
    # now convert and save graph in correct format
    ggC = create_gml.graph2gml(graphW,pos,path,epc=None)
    return ggC
###############################################################################
#
# create gml version of graph for chosen data
#
###############################################################################

graphTagg = pd.read_pickle(overpath+'/noise001/data_test_other/graph_for_DeFiNe_001.gpickle')
posL = pd.read_pickle(overpath+'/noise001/data_test_other/pos_for_DeFiNe_001.gpickle')

for s in range(len(posL)):
    graphD = graphTagg[s]
    posD = posL[s]
    
    pickle.dump(posD, open(overpath+'/others_code/DeFiNe/noise001/pos/{0}_posL.gpickle'.format(s), 'wb'))
    test = create_graphgml(graphD,overpath+"/others_code/DeFiNe/noise001/gml/img_{0}.gml".format(s),posD)
    
    