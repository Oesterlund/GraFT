
###################################################### imports

import collections
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import scipy as sp
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import scipy.stats
import sys
import cvxopt
import cvxopt.glpk
import os
import time
import pandas as pd

overpath = 'define_this_path'
path=overpath+"/others_code/"
sys.path.append(path)

import DeFiNe
import DeFiNe.help

###################################################### calc: program

def calc(self,gtk,inp,sampling,overlap,quality,objective,angle,posL):

    #if(not images[t][e][i].lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))):
    #message = gtk.MessageDialog(parent=self.window,type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK,message_format='Wrong image format.')
    #message.format_secondary_text('Path: '+dir_err+'\n\nRequired: jpg, tiff, or png.\nFound: '+images[t][e][i].split('.')[-1])
    #result=message.run()
    #message.destroy()
    #if result == gtk.RESPONSE_OK: #return 0

    # inp='graph.gml'
    # sampling=1
    # overlap=0
    # quality=0
    # objective=0
    # angle=60.0

    #%%############################# read input

    if self:
        self.builder.get_object('progressbar1').set_text('Reading .gml file.')
        self.builder.get_object('progressbar1').set_fraction(0.25)
        while gtk.events_pending(): gtk.main_iteration()

    gg,pos1,auto,manu=DeFiNe.help.gml2graph(inp)
    pos=posL
    pod = {i: vec for i, vec in enumerate(pos[:,:2])}
    random.shuffle(manu)
    
    #pathm=DeFiNe.help.pathe2pathn(gg,manu)
    #roughs=np.array([DeFiNe.help.path_roughs(p,gg,pos) for p in pathm])
    #angles=np.array([DeFiNe.help.path_angles(p,gg,pos) for p in pathm])
    #qualim=np.vstack([roughs.T,angles.T]).T
	
    # ############################# generate paths

    if self:
        self.builder.get_object('progressbar1').set_text('Generating input paths.')
        self.builder.get_object('progressbar1').set_fraction(0.50)
        while gtk.events_pending(): gtk.main_iteration()

    angle=int(angle)
    pathn=DeFiNe.help.generate_paths(gg,pos,10,angle,mode=sampling)
    roughs=np.array([DeFiNe.help.path_roughs(p,gg,pos) for p in pathn])
    angles=np.array([DeFiNe.help.path_angles(p,gg,pos) for p in pathn])
    qualia=np.vstack([roughs.T,angles.T]).T
    idx=(qualia[:,0]>0)*(qualia[:,10]<angle)
    qualia=qualia[idx]
    patha=[p for i,p in enumerate(pathn) if idx[i]]
    pathe=DeFiNe.help.pathn2pathe(gg,patha)

    ############################## run FCP

    if self:
        self.builder.get_object('progressbar1').set_text('Solving filament cover problem.')
        self.builder.get_object('progressbar1').set_fraction(0.75)
        while gtk.events_pending(): gtk.main_iteration()

    quali=[4,5,1,10][quality]
    if(objective==0):
        xx,obj=DeFiNe.help.setcover(pathe,qualia[:,quali],exact=1-overlap)
    else:
        xx,obj=DeFiNe.help.setcover_mean(pathe,qualia[:,quali],exact=1-overlap)
    auto=[e for i,e in enumerate(pathe) if xx[i]]
    qualia=qualia[xx]

    sim=DeFiNe.help.partition_similarity(auto,manu,gg=gg)
    mpp=DeFiNe.help.partition_matching(auto,manu)

    ############################## extend graph

    if self:
        self.builder.get_object('progressbar1').set_text('Plotting filament cover and analyzes.')
        self.builder.get_object('progressbar1').set_fraction(1.0)
        while gtk.events_pending(): gtk.main_iteration()

    me=[[] for e in gg.edges()]
    for gi,g in enumerate(manu):
        for ei,e in enumerate(g):
            me[e].append(str(gi)+'-'+str(ei))
    ae=[[] for e in gg.edges()]
    for gi,g in enumerate(auto):
        for ei,e in enumerate(g):
            ae[e].append(str(gi)+'-'+str(ei))
    for ei,e in enumerate(gg.edges(data=True)):
        e[2]['manu']=';'.join(me[ei])
        e[2]['auto']=';'.join(ae[ei])

    #%%############################# generate output

    name=inp[:-4]+'_sampling='+str(sampling)+'_overlap='+str(overlap)+'_quality='+str(quality)+'_objective='+str(objective)+'_angle='+str(angle)
    qualin=['rough_len_lin','rough_len_sq','rough_len_euc','rough_len_conv','rough_diff_pair','rough_diff_all','rough_abs_avg','rough_abs_cv','angle_len_lin','angle_diff_mean','angle_diff_max','angle_diff_cv','angle_abs_median','edge_assignment']
    similarities=['VI','RI','JI']
    colors=['SpringGreen','MediumBlue','DarkOrange']
    titles=['manu','auto (FCP)']
    mpl.rcParams['font.size']=5
    mpl.rcParams['figure.figsize']=8.0,6.0 # default 8.0,6.0
    '''
    ml=[';'.join([str(i) for i in j]) for j in manu]
    mx=np.vstack([qualim.T,ml]).T
    mx=np.vstack([qualin,mx])
    np.savetxt(name+'_manu.csv',mx,fmt='%s',delimiter=',')
    '''
    al=[';'.join([str(i) for i in j]) for j in auto]
    ax=np.vstack([qualia.T,al]).T
    ax=np.vstack([qualin,ax])
    np.savetxt(name+'_auto.csv',ax,fmt='%s',delimiter=',')
    plt.figure()
    plt.clf()
    plt.subplot(1,2,1) # manu
    plt.title(titles[0])
    lbs,eps=DeFiNe.help.filament_label(manu,gg)
    lbc,epc,epm=DeFiNe.help.filament_color(eps,lbs,np.sort(mpp))
    DeFiNe.help.graph2gml(gg,pos1,name+'_manu.gml',epc=epc)
    #nx.draw_networkx(gg,pod,edge_color=epc,width=2,node_size=0,edge_labels=lbc,font_color='black',font_size=5,bbox={'edgecolor':'none','facecolor':'none'})
    #nx.draw_networkx(gg,pod,edge_color=epc,width=2,node_size=0,edge_labels=lbc,font_color='black',font_size=5,bbox={'edgecolor':'none','facecolor':'none'})
    # nx.draw_networkx_edges(gg,pod,edge_color=epc,width=2)
    # nx.draw_networkx_edge_labels(gg,pod,edge_labels=lbc,font_color='black',font_size=5,bbox={'edgecolor':'none','facecolor':'none'})
    plt.ylim(plt.ylim()[::-1])
    plt.axis('off')

    #plt.subplot(1,2,2) # auto
    plt.figure(figsize=(10,10))
    #plt.title(titles[1])
    lbs,eps=DeFiNe.help.filament_label(auto,gg)
    edges,values = zip(*nx.get_edge_attributes(gg,'auto').items())
    lbc,epc,epm=DeFiNe.help.filament_color(eps,lbs,mpp)
    DeFiNe.help.graph2gml(gg,pos1,name+'_auto.gml',epc=epc)
    #nx.draw_networkx(gg,pod,edge_color=epc,width=2,node_size=0,font_color='black',font_size=10,bbox={'edgecolor':'none','facecolor':'none'})
    cmap=plt.get_cmap('tab20',int(np.max(eps[0])))
    nx.draw(gg,pod,edge_cmap=cmap,edge_color=eps[0],node_size=35,width=10,alpha=0.8)
    
    #nx.draw_networkx(gg,pod,edge_color=epc,width=2,node_size=0,edge_labels=lbc,font_color='black',font_size=5,bbox={'edgecolor':'none','facecolor':'none'})
    # nx.draw_networkx_edges(gg,pod,edge_color=epc,width=2)
    # nx.draw_networkx_edge_labels(gg,pod,edge_labels=lbc,font_color='black',font_size=5,bbox={'edgecolor':'none','facecolor':'none'})
    plt.ylim(plt.ylim()[::-1])
    plt.axis('off')
    plt.savefig(name+'DeFiNe_graph.png')
    '''
    plt.subplot(2,4,3) # dist len
    s=0
    mm=max(qualim[:,s].max(),qualia[:,s].max())
    sm=int(int(mm+10)/10*10)
    plt.hist(qualim[:,s],range(sm+10),lw=2,color='black',histtype='step',align='mid',alpha=0.5)
    plt.hist(qualia[:,s],range(sm+10),lw=2,color='black',histtype='step',align='mid',alpha=1.0)
    plt.title('p_KS = '+'%.1e'%sp.stats.ks_2samp(qualim[:,s],qualia[:,s])[1])
    plt.xlim(0,sm)
    plt.xlabel('filament length')
    plt.ylabel('frequency')

    plt.subplot(2,4,4) # dist ang
    a=10
    plt.hist(qualim[:,a],range(0,61,5),lw=2,color='black',histtype='step',align='mid',alpha=0.5)
    plt.hist(qualia[:,a],range(0,61,5),lw=2,color='black',histtype='step',align='mid',alpha=1.0)
    plt.title('p_KS = '+'%.1e'%sp.stats.ks_2samp(qualim[:,a],qualia[:,a])[1])
    plt.xlim(-5,65)
    plt.xlabel('filament angle')
    plt.ylabel('frequency')

    plt.subplot(2,4,7) # sim
    sims=np.reshape(sim[3:],(-1,2)).T
    S=len(sims[0])
    H=max(S,1)
    for i in range(3):
        plt.plot(range(1,H+1),sim[i]*np.ones(H),label=similarities[i],lw=2,color=colors[i],ls='--')
    for i in range(2*min(1,S)):
        plt.plot(range(1,H-1),sims[i][:-2],lw=2,color=colors[1+i],label=similarities[1+i]+'$^d$')
        diff=(sims[i][-1]-sims[i][-2])
        plt.plot([H-2+0.00,H-2+0.25],[sims[i][-2],sims[i][-2]+0.25*diff],lw=2,color=colors[1+i],ls=':')
        plt.plot([H-2+0.75,H-2+1.00],[sims[i][-2]+0.75*diff,sims[i][-1]],lw=2,color=colors[1+i],ls=':')
        plt.plot([H-1],sims[i][-1:],lw=2,color=colors[1+i],marker='o')
    plt.xlabel('edge distance d')
    plt.ylabel('partition similarity')
    plt.ylim(-0.05,1.05)
    plt.xlim(1,H-1)
    plt.xticks(np.hstack([range(1,H-1,2),H-1]),np.hstack([range(1,H-1,2),'$\\infty$']))
    plt.legend(loc=0,frameon=0)

    plt.subplot(2,4,8) # corr
    r,a=6,10
    man=qualim[:,0]>1
    if(man.sum()==0):
        man=qualim[:,0]>-99
    poly=np.polyfit(qualim[:,r][man],qualim[:,a][man],1)
    plt.plot(qualim[:,r],np.poly1d(poly)(qualim[:,r]),lw=2,color='black',alpha=0.5)
    plt.plot(qualim[:,r][man],qualim[:,a][man],marker='s',ls='',mew=2,mfc='none',mec=[0,0,0,0.5],alpha=0.5,ms=6,label='manu')
    plt.plot(qualim[:,r][~man],qualim[:,a][~man],marker='s',ls='',mew=2,mfc='none',mec=[0,0,0,0.5],alpha=0.5,ms=2)
    aan=qualim[:,0]>1
    if(aan.sum()==0):
        aan=qualim[:,0]>-99
    poly=np.polyfit(qualim[:,r][aan],qualim[:,a][aan],1)
    plt.plot(qualim[:,r],np.poly1d(poly)(qualim[:,r]),lw=2,color='black',alpha=1.0)
    plt.plot(qualim[:,r][aan],qualim[:,a][aan],marker='o',ls='',mew=2,mfc='none',color=[0,0,0,1.0],alpha=1.0,ms=6,label='auto')
    plt.plot(qualim[:,r][~aan],qualim[:,a][~aan],marker='o',ls='',mew=2,mfc='none',color=[0,0,0,1.0],alpha=1.0,ms=2)
    plt.title('auto p_tau = '+'%.1e'%sp.stats.kendalltau(qualim[:,r],qualim[:,a])[1]+'\n manu p_tau = '+'%.1e'%sp.stats.kendalltau(qualim[:,r],qualim[:,a])[1],ha='right')
    plt.legend(loc=0,frameon=0)
    plt.xlabel('filament weight')
    plt.ylabel('filament angle')
	'''
    plt.tight_layout(pad=0.1)
    plt.savefig(name+'.pdf')
    plt.savefig(name+'.svg')
    #plt.show()

    if self:
        self.builder.get_object('progressbar1').set_text('Done.')
        self.builder.get_object('progressbar1').set_fraction(0.0)
        while gtk.events_pending(): gtk.main_iteration()

    #%%############################# end script

    return gg

###############################################################################
#
# density
#
###############################################################################
pathgml = '/home/isabella/Documents/PLEN/dfs/others_code/DeFiNe/noise001/gml/'
pathsave = '/home/isabella/Documents/PLEN/dfs/others_code/DeFiNe/noise001/graphs/'

for m in range(8,100):
    start_time = time.time()
    inp = pathgml + 'img_{0}.gml'.format(m)
    sampling=1
    overlap=1
    quality=0
    objective=0
    angle=60.0
    posL = pd.read_pickle('/home/isabella/Documents/PLEN/dfs/others_code/DeFiNe/noise001/pos/{0}_posL.gpickle'.format(m))
    graph=calc(None,None,inp,sampling,overlap,quality,objective,angle,posL)

    
    end_time = time.time()
    duration = end_time - start_time

list(graph.edges(data='auto'))
test = np.asarray(list(graph.edges(data='auto')))[:,2]
tags=[int(test[i].split('-')[0]) for i in range(len(test))]
len(np.unique(tags))


