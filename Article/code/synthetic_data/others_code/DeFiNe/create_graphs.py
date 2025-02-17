#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:46:01 2022

@author: pgf840
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import sys

overpath = 'define_this_path'
path2=overpath+"/others_code/DeFiNe/"
sys.path.append(path2)
import create_gml

plt.close('all')


###############################################################################
#
# create gml version of graph for chosen data
#
###############################################################################

graphTagg = pd.read_pickle('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/data_to_test_against_other/graph_for_DeFiNe.gpickle')
posL = pd.read_pickle('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/data/data_to_test_against_other/pos_for_DeFiNe.gpickle')

for s in range(len(posL)):
    graphD = graphTagg[s]
    posD = posL[s]
    
    create_gml.create_graphgml(graphD,"/home/isabella/Documents/PLEN/dfs/others_code/DeFiNe/gml/img_{0}.gml".format(s),posD)
    