#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pgf840
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def color2hex(col):
    return ('#%02x%02x%02x'%tuple(np.multiply(col[:3],255).astype(int))).replace('f','F')


def graph2gml(gg,pos,path,epc=None):
    ew=1.0*np.array([d['weight'] for u,v,d in gg.edges(data=True)])
    ew=ew/ew.max()
    ec=plt.cm.jet(ew)
    ly,lx=pos.shape
    pox=np.zeros((ly,3))
    pox[:,:lx]=pos
    gg.graph['directed']=0
    gg.graph['defaultnodesize']=0
    for i,n in enumerate(gg.nodes(data=True)):
       n[1]['graphics']={'x':pox[i][0],'y':pox[i][1],'z':pox[i][2],'hasFill':0,'hasOutline':0}
    
    for i,e in enumerate(gg.edges(data=True)):
        e[2]['graphics']={'targetArrow':'none','width':5.0*ew[i],'fill':color2hex(ec[i])}

    if(epc!=None):
        for i,e in enumerate(gg.edges(data=True)):
            e[2]['graphics']={'targetArrow':'none','width':5.0*ew[i],'fill':color2hex(epc[i])}
    nx.write_gml(gg,path)
    return gg

def strG(graphW):
    grapCur=nx.Graph()
    for node1, node2, property in graphW.edges(data=True):
        edist = property['edist']
        fdist = property['fdist']
        weight = property['weight']
        capa = property['capa']
        lgth = property['lgth']
        conn = property['conn']
        #jump = property['jump']
        manu = property['manu']
        auto = property['auto']
        grapCur.add_edge(str(node1), str(node2), edist=edist, fdist=float(fdist), weight=weight, capa=capa, lgth=lgth, conn=conn, manu=manu,auto=auto)
    return(grapCur)

def create_graphgml(graph, path,pos):
    g3=graph.copy()
    simpleGraph = nx.Graph(g3)
    edges= list(simpleGraph.edges())
    tag=0
    for i in range(len(simpleGraph.edges())):
        simpleGraph[edges[i][0]][edges[i][1]]['manu'] = '{}-{}'.format(tag,0)
        simpleGraph[edges[i][0]][edges[i][1]]['auto'] = '{}-{}'.format(tag,0)
        tag+=1
      
    graphW = strG(simpleGraph)
    # now convert and save graph in correct format
    ggC = graph2gml(graphW,pos,path,epc=None)
    return ggC
