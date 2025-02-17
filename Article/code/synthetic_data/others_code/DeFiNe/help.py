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

###################################################### help: gml

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
    if np.any(epc!='None'):
        for i,e in enumerate(gg.edges(data=True)):
            e[2]['graphics']={'targetArrow':'none','width':5.0*ew[i],'fill':color2hex(epc[i])}
    nx.write_gml(gg,path)
    return None

def gml2graph(path):
    gg1=nx.read_gml(path,destringizer=int)
    E=gg1.number_of_edges()
    N=gg1.number_of_nodes()
    pos1=np.zeros((N,3))
    auto1=[range(E)]
    manu1=[range(E)]
    NS=gg1.nodes(data=True)
    ES=gg1.edges(data=True)
    NK=NS[0]['graphics'].keys()
    EK=list(ES)[0][2].keys()

    if('x' in NK):
        pos1[:,0]=[n[1]['graphics']['x'] for n in NS]
    if('y' in NK):
        pos1[:,1]=[n[1]['graphics']['y'] for n in NS]
    if('z' in NK):
        pos1[:,2]=[n[1]['graphics']['z'] for n in NS]

    if('auto' in EK):

        lists=np.vstack([np.vstack([[j for j in i.split('-')] for i in e[2]['auto'].split(',')]) for e in ES]).astype('int')
        M=max(lists[:,0])+1
        auto1=[[] for m in range(M)]
        for m in range(M):
            K=max(lists[lists[:,0]==m,1])+1
            auto1[m]=[[] for k in range(K)]
        for ei,e in enumerate(ES):
            idx=np.vstack([[j for j in i.split('-')] for i in e[2]['auto'].split(',')]).astype('int')
            for i in idx:
                auto1[i[0]][i[1]]=ei

    if('manu' in EK):

        lists=np.vstack([np.vstack([[j for j in i.split('-')] for i in e[2]['manu'].split(',')]) for e in ES]).astype('int')
        M=max(lists[:,0])+1
        manu1=[[] for m in range(M)]
        for m in range(M):
            K=max(lists[lists[:,0]==m,1])+1
            manu1[m]=[[] for k in range(K)]
        for ei,e in enumerate(ES):
            idx=np.vstack([[j for j in i.split('-')] for i in e[2]['manu'].split(',')]).astype('int')
            for i in idx:
                manu1[i[0]][i[1]]=ei

    return gg1,pos1,auto1,manu1

###################################################### help: partitions

def partition_graph(aa,bb,gg):

    E=gg.number_of_edges()
    gl=nx.line_graph(gg)
    lbl={}
    for i,g in enumerate(gg.edges()):
        lbl[g]=i
    gl=nx.relabel_nodes(gl,lbl)

    ad=dict(nx.all_pairs_shortest_path_length(gl))
    am=np.zeros((E,E))
    for ni,n in enumerate(gl.nodes()):
        for mi,m in enumerate(gl.nodes()):
            if(n in ad.keys()):
                if(m in ad[n].keys()):
                    am[ni,mi]=ad[n][m]
                    am[mi,ni]=ad[n][m]
    M=int(am.max()+0.5)+1
    am[am==0]=M
    hh=np.zeros((M,2,2))
    for e0 in range(E):
        for e1 in range(e0):
            xa=np.max([(e0 in i)+(e1 in i) for i in aa])
            xb=np.max([(e0 in i)+(e1 in i) for i in bb])
            di=int(am[e0,e1])-1
            if(xa==2 and xb==2):
                hh[di,0,0]+=1.0
            elif(xa<2 and xb==2):
                hh[di,1,0]+=1.0
            elif(xa==2 and xb<2):
                hh[di,0,1]+=1.0
            elif(xa<2 and xb<2):
                hh[di,1,1]+=1.0

    return hh

def partition_similarity(aa,bb,gg=None):

    # aa,bb=auto,manu
    uu=set(np.hstack(bb))
    E=max(uu)+1
    aa=[set(a) for a in aa]
    bb=[set(b) for b in bb]
    A=len(aa)
    B=len(bb)
    U=len(uu)
    nn=np.zeros((A+1,B+1))
    for ai,a in enumerate(aa):
        for bi,b in enumerate(bb):
            nn[ai,bi]=len(a.intersection(b))
    nn[A,:]=np.sum(nn,0)
    nn[:,B]=np.sum(nn,1)
    pa=nn[:A,B]
    pb=nn[A,:B]
    pc=nn[:A,:B]
    meila=1.0+1.0/U/np.log(U)*np.sum((pc*np.log(pc.T/pa).T+pc*np.log(pc/pb))[pc>0])

    hh=np.zeros((2,2))
    for e0 in range(E):
        for e1 in range(e0):
            xa=np.max([(e0 in i)+(e1 in i) for i in aa])
            xb=np.max([(e0 in i)+(e1 in i) for i in bb])
            if(xa==2 and xb==2):
                hh[0,0]+=1.0
            elif(xa<2 and xb==2):
                hh[1,0]+=1.0
            elif(xa==2 and xb<2):
                hh[0,1]+=1.0
            elif(xa<2 and xb<2):
                hh[1,1]+=1.0
    ha,hb,hc,hd=hh.flatten()
    rand=(ha+hd)/(ha+hb+hc+hd)
    jaccard=ha/(ha+hb+hc)

    if(gg==None):
        sim=[]

    else:

        hh=partition_graph(aa,bb,gg)
        sim=np.zeros(2*len(hh))
        for hi,h in enumerate(hh):
            ha,hb,hc,hd=np.sum(hh[:hi+1],0).flatten()
            sim[2*hi+0]=(ha+hd)/(ha+hb+hc+hd)
            sim[2*hi+1]=ha/(ha+hb+hc)

    return np.hstack([[meila,rand,jaccard],sim])

def partition_matching(aa,bb):

    la,lb=len(aa),len(bb)
    dd=np.zeros((la,lb))
    for ia,a in enumerate(aa):
        for ib,b in enumerate(bb):
            dd[ia,ib]=len(set(a).intersection(set(b)))
    la,lb=dd.shape
    N=max(la,lb)
    cc=-np.ones((N,N))
    cc[:la,:lb]=dd
    c=cc.flatten()
    #G=-np.diag(np.ones(N*N))
    G2,G0,G1=[],[],[]
    for n in range(N*N):
        G2.append(-1)
        G0.append(n)
        G1.append(n)
    h=np.zeros((N*N,1))
    #A=np.zeros((N+N,cc.size))
    A2,A0,A1=[],[],[]
    for a in range(N):
        for b in range(N):
            A2.append(1)
            A0.append(0*N+a)
            A1.append(a*N+b)
            A2.append(1)
            A0.append(1*N+b)
            A1.append(a*N+b)
#            A[0*N+a,a*N+b]=1
#            A[1*N+b,a*N+b]=1
    b=np.ones((N+N,1))
    sol=cvxopt.glpk.lp(c=cvxopt.matrix(-c),G=cvxopt.spmatrix(G2,G0,G1,(N*N,N*N)),h=cvxopt.matrix(h),A=cvxopt.spmatrix(A2,A0,A1,(N+N,N*N)),b=cvxopt.matrix(b))
    yy=np.reshape([s for s in sol[1]],cc.shape)>0
    mp=np.where(yy)[1]
    return mp

###################################################### help: quality

def path_roughs(path,gg,pos):
    ec=[]
    edges=list(gg.edges())
    s3=0.0
    for pi,(p0,p1) in enumerate(zip(path[:-1],path[1:])):
        i0,i1=int(p0),int(p1)
        s3+=np.sqrt(np.sum((pos[i0]-pos[i1])**2))
        if(p1 in gg[p0].keys()):
            ec.append(gg[p0][p1]['capa'])
        else:
            ec.append(gg[p1][p0]['capa'])
    s1=len(ec)
    s2=np.sqrt(s1)
    s4=s3/np.max([np.max([np.max(np.abs(pos[int(n)]-pos[int(m)])) for n in path]) for m in path])
    if(s1>1):
        s5=np.mean(np.abs(np.diff(ec)))
        s6=1.0*(np.max(ec)-np.min(ec))/s1
        s7=np.mean(ec)
        s8=np.std(ec)/np.mean(ec)
    else:
        s5,s6,s7,s8=np.min(ec)*np.ones(4)
    return s1,s2,s3,s4,s5,s6,s7,s8#len_lin,len_sq,len_euc,len_conv,diff_pair,diff_all,abs_avg,abs_cv,

def angle180edges(duv,dvw):
    return np.arccos(np.dot(duv,dvw)/sp.linalg.norm(duv)/sp.linalg.norm(dvw))*180.0/np.pi

def angle360smooth(angles):
    dangle=np.arange(-3.0,3.1,1.0)*360.0
    for i in range(len(angles)-1):
        qi=np.argmin(np.abs(angles[i+1]+dangle-angles[i]))
        angles[i+1]+=dangle[qi]
    return angles

def angle360xy(dxy):
    dx,dy=dxy[:2]
    return np.mod(np.arctan2(dx,dy)*180.0/np.pi,360.0)

def path_angles(path,gg,pos):
    P=len(path)
    edges=list(gg.edges())
    ang=[]
    for p in range(P-2):
        u,v,w=path[p:p+3]
        duv=pos[int(u)]-pos[int(v)]
        dvw=pos[int(v)]-pos[int(w)]
        ang.append(angle180edges(duv,dvw))
    a1=len(ang)
    if(a1>0):
        a2=np.mean(ang)
        a3=np.max(ang)
        a4=np.std(ang)/np.mean(ang)
    else:
        a2,a3,a4=0.0*np.ones(3)
    angles=[]
    for p in range(P-1):
        p0,p1=int(path[p+0]),int(path[p+1])
        angles.append(angle360xy(pos[p1]-pos[p0]))
    a5=np.mod(np.median(angle360smooth(angles)),180.0)
    return a1,a2,a3,a4,a5#len_lin,diff_mean,diff_max,diff_cv,abs_median

###################################################### help: paths

def random_spanning_tree(gg):
    for u,v,d in gg.edges(data=True):
        d['rand']=1.0*np.random.rand()
    gm = nx.minimum_spanning_tree(gg,weight='rand')
    return gm

def bfs_paths_angles(gg,pos,thres,start,goal):
    gx=nx.to_dict_of_dicts(gg)
    for i in gx:
        gx[i]=set(gx[i].keys())
    queue=[(start,[start])]
    while queue:
        (vertex,path)=queue.pop(0)
        for next in gx[vertex]-set(path[-4:]):
            pnxt=path+[next]
            if next==goal:
                yield pnxt
            elif(max(collections.Counter(pnxt).values())<3):
                if(path_angles(pnxt,gg,pos)[2]<thres):
                    queue.append((next,pnxt))

def generate_paths(gg,pos,sample,thres,mode=0):

    if(mode==0):
        N=gg.number_of_nodes()
        paths=[]
        for n in range(N):
            print('BFS',n,N)
            for m in range(n):
                patha=list(bfs_paths_angles(gg,pos,thres,n,m))
                for path in patha:
                    if(not path in paths and len(path)>1):
                        paths.append(path)

    else:
        paths=[]
        for i in range(sample):
            print('RMST',i,sample)
            gm=random_spanning_tree(gg)
            pathd=nx.all_pairs_dijkstra_path(gm)
            for idp, pathi in pathd:
                for m in pathi.keys():
                    path=pathi[m]
                    if(not path in paths and len(path)>1):
                        paths.append(path)
    return paths

###################################################### help: scp

def setcover(sets,cost,exact=0):

    uni=list(set(np.hstack(sets)))
    S=len(sets)
    U=len(uni)
    B=set(range(S))
    SS=range(S)
    UU=range(U)

    c=cost
    h=-np.ones((S+S+U,1))
    h[0:S]=0
    h[S:S+S]=2
    #G=np.zeros((U,S))
    G2,G0,G1=[],[],[]
    for s in SS:
        G2.append(-1)
        G0.append(s)
        G1.append(s)
        G2.append(1)
        G0.append(S+s)
        G1.append(s)
    for u in UU:
        for s in SS:
            if(uni[u] in sets[s]):
                #G[u,s]=-1
                G2.append(-1)
                G0.append(S+S+u)
                G1.append(s)

    if(exact):
        h=np.vstack([h,-h[-U:]])
        #G=np.vstack([G,-G])
        for u in UU:
            for s in SS:
                if(uni[u] in sets[s]):
                    #G[u,s]=-1
                    G2.append(1)
                    G0.append(S+S+U+u)
                    G1.append(s)
        U=U+U

    res=cvxopt.glpk.ilp(c=cvxopt.matrix([i for i in c]),G=cvxopt.spmatrix(G2,G0,G1,(S+S+U,S)),h=cvxopt.matrix(h),B=B)
    xx=np.array([r>0 for r in res[1]])
    return xx,np.sum([cost[s] for s in SS if xx[s]>0])

def setcover_mean(sets,cost,exact=0):

    uni=list(set(np.hstack(sets)))
    S=len(sets)
    U=len(uni)
    B=set(range(S))
    SS=range(S)
    UU=range(U)

    c=np.hstack([cost*0,cost*1,0])
    h=np.zeros((S+S+1+2+S+S+S+S+S+1+U,1))
    h[S+S+1+0]=1                            # sum z <= 1
    h[S+S+1+1]=-1                           # sum z >= 1
    h[S+S+1+2+S+S:S+S+1+2+S+S+S]=2          # y-z+kx <= K
    h[S+S+1+2+S+S+S:S+S+1+2+S+S+S+S+S+1]=2  # x,z,y <= 2
    #G=np.zeros((S+S+1+2+S+S+S+U,S+S+1))
    G2,G0,G1=[],[],[]
    for s in range(S+S+1):
        #G[s,s]=-1
        G2.append(-1)
        G0.append(s)
        G1.append(s)
    for s in range(S):
        #G[S+S+1+0,S+s]=1
        G2.append(1)
        G0.append(S+S+1+0)
        G1.append(S+s)
        #G[S+S+1+1,S+s]=-1
        G2.append(-1)
        G0.append(S+S+1+1)
        G1.append(S+s)
        #G[S+S+1+2+s,S+s]=1
        G2.append(1)
        G0.append(S+S+1+2+s)
        G1.append(S+s)
        #G[S+S+1+2+s,S+S]=-1
        G2.append(-1)
        G0.append(S+S+1+2+s)
        G1.append(S+S)
        #G[S+S+1+2+S+s,s]=-2
        G2.append(-2)
        G0.append(S+S+1+2+S+s)
        G1.append(s)
        #G[S+S+1+2+S+s,S+s]=1
        G2.append(1)
        G0.append(S+S+1+2+S+s)
        G1.append(S+s)
        #G[S+S+1+2+S+S+s,s]=2
        G2.append(2)
        G0.append(S+S+1+2+S+S+s)
        G1.append(s)
        #G[S+S+1+2+S+S+s,S+s]=-1
        G2.append(-1)
        G0.append(S+S+1+2+S+S+s)
        G1.append(S+s)
        #G[S+S+1+2+S+S+s,S+S]=1
        G2.append(1)
        G0.append(S+S+1+2+S+S+s)
        G1.append(S+S)
    for s in range(S+S+1):
        G2.append(1)
        G0.append(S+S+1+2+S+S+S+s)
        G1.append(s)
    for u in UU:
        #G[S+S+1+2+S+S+S+u,S+S]=1
        G2.append(1)
        G0.append(S+S+1+2+S+S+S+S+S+1+u)
        G1.append(S+S)
        for s in SS:
            if(uni[u] in sets[s]):
                #G[S+S+1+2+S+S+S+u,S+s]=-1
                G2.append(-1)
                G0.append(S+S+1+2+S+S+S+S+S+1+u)
                G1.append(S+s)

    if(exact):
        h=np.vstack([h,-h[-U:]])
        #G=np.vstack([G,-G[-U:]])
        for u in UU:
            #G[S+S+1+2+S+S+S+u,S+S]=1
            G2.append(-1)
            G0.append(S+S+1+2+S+S+S+S+S+1+U+u)
            G1.append(S+S)
            for s in SS:
                if(uni[u] in sets[s]):
                    #G[S+S+1+2+S+S+S+u,S+s]=-1
                    G2.append(1)
                    G0.append(S+S+1+2+S+S+S+S+S+1+U+u)
                    G1.append(S+s)
        U=U+U

    res=cvxopt.glpk.ilp(c=cvxopt.matrix(c),G=cvxopt.spmatrix(G2,G0,G1,(S+S+1+2+S+S+S+S+S+1+U,S+S+1)),h=cvxopt.matrix(h),B=B)
    xx=np.array([r for r in res[1]])

    return xx[:S]>0,np.sum([cost[s] for s in range(S) if xx[S+s]>0])*xx[S+S]

###################################################### help: label

def filament_label(partition,gg):
    ps=list(np.hstack(partition))
    cs=np.max([ps.count(i) for i in set(ps)])
    E=gg.number_of_edges()
    eps=np.ones((cs,E))*np.nan
    lbs={}
    for ei,e in enumerate(gg.edges()):
        pis=np.array([pi for pi,p in enumerate(partition) if ei in p])
        for pi,p in enumerate(pis):
            eps[pi][ei]=p
        lbs[tuple(e)]=','.join([str(int(i)) for i in eps[:,ei] if i==i])
    return lbs,eps

def filament_color(eps,lbs,mp):
    C=len(mp)
    E=len(eps[0])
    ept=copy.deepcopy(eps)
    ept[~np.isfinite(eps)]=0
    ept=ept.astype('int')
    epm=mp[ept].astype('float')
    epm[~np.isfinite(eps)]=np.nan
    epc=np.zeros((E,4))
    for i,e in enumerate(epm.T):
        epc[i]=np.nanmean(plt.cm.tab20(1.0*e[e==e]/(C-1.0)),0)
    lbm=copy.deepcopy(lbs)
    for l in lbm.keys():
        lbm[l]=','.join([str(mp[int(i)]) for i in lbm[l].split(',')])
    return lbm,epc,epm

###################################################### help: conversion

def pathn2pathe(gg,pathx):
    edges=np.sort(gg.edges(),1)
    pathe=[]
    for i,p in enumerate(pathx):
        path=[]
        for pi in np.sort(np.array(tuple(zip(p[:-1],p[1:]))),1):
            path.append(np.argmax(np.sum(edges==pi,1)==2))
        pathe.append(path)
    return pathe

def pathe2pathn(gg,manu):
    edges=list(gg.edges())
    pathn=[]
    for g in manu:
        start=[e for ei,e in enumerate(edges[g[0]]) if not e in edges[g[1]]][0]
        path=[start]
        for i in range(len(g)):
            nxt=[e for ei,e in enumerate(edges[g[i]]) if e!=path[-1]][0]
            path.append(nxt)
        pathn.append(path)
    return pathn
