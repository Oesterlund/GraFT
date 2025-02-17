#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import pandas as pd
import pickle

from graft import utilsF


def create_output_dirs(output_dir):
    """Ensure that the given output directory incl. subdirectories exists."""
    for subdir_name in ('n_graphs', 'circ_stat', 'mov', 'plots'):
        subdir_path = os.path.join(output_dir, subdir_name)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)


def generate_default_mask(image_shape):
    """Generate a default mask of ones based on the image shape."""
    if len(image_shape) == 3:  # Time-series image
        return np.ones(image_shape[1:])
    elif len(image_shape) == 2:  # Still image
        return np.ones(image_shape)
    else:
        raise ValueError("Unsupported image shape. Expected 2 or 3 dimensions.")


def pad_timeseries_images(img_o):
    M,N,P = (img_o.shape)
    imgP=np.zeros((M,N+2,P+2))

    for m in range(len(img_o)):
        imgP[m] = np.pad(img_o[m], 1, 'constant')
    return imgP

def process_individual_image(image, output_dir, size, eps, thresh_top, sigma, small, angleA, overlap, index=None):
    # create graph
    graph_s, posL, imgSkel, imgAF, imgBl, imF, mask, df_pos = utilsF.creategraph(image, size=size, eps=eps, thresh_top=thresh_top, sigma=sigma, small=small)

    # find all dangling edges and mark them
    graphD = utilsF.dangling_edges(graph_s.copy())

    # create line graph
    lgG = nx.line_graph(graph_s.copy())

    # calculate the angles between two edges from the graph that is now
    # represented by edges in the line graph
    lgG_V = utilsF.lG_edgeVal(lgG.copy(), graphD, posL)

    # run depth first search
    graphTagg = utilsF.dfs_constrained(graph_s.copy(), lgG_V.copy(), imgBl, posL, angleA, overlap)

    utilsF.draw_graph_filament_nocolor(image, graphTagg, posL, "", 'filament')
    filename =  f'graph.png' if index is None else f'graph{index}.png'
    plt.savefig(os.path.join(output_dir, 'n_graphs', filename))
    plt.close('all')

    no_filaments = len(np.unique(np.asarray(list(graphTagg.edges(data='filament')))[:,2]))
    print('filament defined: ', no_filaments)
    return posL, graphTagg, imF


def tag_graphs(img_o, graphTagg, posL, max_cost, memKeep, output_dir):
    # first graph needs unique tags
    for node1, node2, property in graphTagg[0].edges(data=True):
        for n in range(len(graphTagg[0][node1][node2])):
            graphTagg[0][node1][node2][n]['tags'] = property['filament']

    max_tag = np.max(list(graphTagg[0].edges(data='filament')),axis=0)[2]

    g_tagged = [0]*(len(img_o))
    g_tagged[0] = graphTagg[0]
    cost = [0]*(len(img_o)-1)
    tag_new = [0]*(len(img_o))
    tag_new[0] = max_tag
    filamentsNU = []

    for i in range(len(img_o)-1):
        g_tagged[i+1],cost[i],tag_new[i+1],filamentsNU = \
            utilsF.filament_tag(g_tagged[i], graphTagg[i+1],
                                posL[i], posL[i+1], tag_new[i],
                                max_cost, filamentsNU, memKeep)

    pickle.dump(g_tagged, open(os.path.join(output_dir, 'tagged_graph.gpickle'), 'wb'))
    return g_tagged, tag_new


def setup_plot(figsize=(10, 10), x_label='', y_label='', label_size=24):
    plt.figure(figsize=figsize)
    plt.xlabel(x_label, size=label_size)
    plt.ylabel(y_label, size=label_size)
    plt.rc('xtick', labelsize=label_size)
    plt.rc('ytick', labelsize=label_size)


def save_and_close_plot(path):
    plt.savefig(path)
    plt.close('all')


def create_histogram(data, path, bins=20, density=False, color='green', title=''):
    counts, bins = np.histogram(list(data), bins, density=density)
    setup_plot((10, 7), 'frames', title)
    plt.hist(bins[:-1], bins, weights=counts, color=color)
    save_and_close_plot(path)


def create_scatter_plot(x_data, y_data, path, title='', x_label='frames', y_label=''):
    setup_plot(x_label=x_label, y_label=y_label)
    plt.scatter(x_data, y_data)
    save_and_close_plot(path)


def create_all(pathsave, img_o, maskDraw, size, eps, thresh_top, sigma,
               small, angleA, overlap, max_cost, name_cell,
               parallelize=True):
    create_output_dirs(pathsave)

    posL = [0]*len(img_o)
    imF = [0]*len(img_o)
    graphTagg = [0]*len(img_o)

    imgP = pad_timeseries_images(img_o)

    if parallelize:  # prcoess images in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_individual_image, image, pathsave, size, eps, thresh_top, sigma, small, angleA, overlap, i): i for i, image in enumerate(imgP)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                posL[i], graphTagg[i], imF[i] = future.result()
    else:  # process images sequentially
        for i, image in enumerate(imgP):
            posL[i], graphTagg[i], imF[i] = process_individual_image(image, pathsave, size, eps, thresh_top, sigma, small, angleA, overlap, i)

    pickle.dump(posL, open(os.path.join(pathsave, 'posL.gpickle'), 'wb'))

    if(len(img_o)<20):
        memKeep = len(img_o)
    else:
        memVal = 20
        memKeep = utilsF.signMem(graphTagg[0:memVal],posL[0:memVal])

    g_tagged, tag_new = tag_graphs(img_o, graphTagg, posL, max_cost, memKeep, pathsave)

    for i, _image in enumerate(img_o):
        utilsF.draw_graph_filament_track_nocolor(imgP[i], g_tagged[i], posL[i], f"graph {i+1}", max(tag_new), padv=50)
        save_and_close_plot(os.path.join(pathsave, "mov", f"trackgraph{i+1}.png"))

    # filaments per frame plot
    unique_filaments = [len(np.unique(np.asarray(list(g.edges(data='tags')))[:, 2])) for g in g_tagged]
    create_scatter_plot(np.arange(0, len(unique_filaments)), unique_filaments, os.path.join(pathsave, 'plots', 'filaments_per_frame.png'), y_label='# filaments')

    pd_fil_info = utilsF.filament_info_time(imgP, g_tagged, posL, pathsave, imF, maskDraw)
    pd_fil_info.to_csv(os.path.join(pathsave, 'tracked_filaments_info.csv'), index=False)

    # filaments survival plot
    filament_counts = pd_fil_info['filament'].value_counts()
    create_histogram(filament_counts, os.path.join(pathsave, 'plots', 'survival_filaments.png'), title='filaments survival')
    create_histogram(filament_counts, os.path.join(pathsave, 'plots', 'survival_filaments_normalized.png'), density=True, title='filaments survival')

    # filament density plot
    dens = np.zeros(len(img_o))
    fil_len = np.zeros(len(img_o))
    fil_I = np.zeros(len(img_o))
    for i in range(len(img_o)):
        dens[i] = pd_fil_info[pd_fil_info['frame number']==i]['filament density'].values[0]
        fil_len[i] =np.median(pd_fil_info[pd_fil_info['frame number']==i]['filament length'])
        fil_I[i] = np.median(pd_fil_info[pd_fil_info['frame number']==i]['filament intensity per length'])
    create_scatter_plot(np.arange(len(img_o)), dens, os.path.join(pathsave, 'plots', 'filament_density.png'), y_label='filament density')

    # filament length plot
    create_scatter_plot(np.arange(len(img_o)), fil_len, os.path.join(pathsave, 'plots', 'filamentlength.png'), y_label='filament median length')

    # circular mean angle plot
    mean_angle,var_val = utilsF.circ_stat_plot(pathsave,pd_fil_info)
    setup_plot((10, 10), 'Frames', 'Circular mean angle')
    plt.scatter(np.arange(len(mean_angle)), mean_angle)
    plt.plot(np.arange(len(mean_angle)), np.full(len(mean_angle), np.mean(mean_angle)), color='black', linestyle='dashed')
    save_and_close_plot(os.path.join(pathsave, 'plots', 'angles_mean.png'))

    # circular variance of angles plot
    create_scatter_plot(np.arange(len(var_val)), var_val, os.path.join(pathsave, 'plots', 'angles_var.png'), y_label='circular variance of angles')

    # survival length plot for each unique filament
    setup_plot((10, 10), 'Survival frames', 'filament length')
    for s in pd_fil_info['filament'].unique():
        fil = pd_fil_info[pd_fil_info['filament'] == s]['filament length'].values
        plt.plot(np.arange(len(fil)), fil)
    save_and_close_plot(os.path.join(pathsave, 'plots', 'survival_len.png'))

    ###############################################################################
    #
    # mean/median value per frame
    #
    ###############################################################################

    df_angles2 = pd.DataFrame()
    df_angles2['angles'] = mean_angle
    df_angles2['var'] = var_val
    df_angles2['frame density'] = dens
    df_angles2['filament median length'] = fil_len
    df_angles2['filament mediam intensity per length'] = fil_I
    df_angles2['name'] = name_cell

    df_angles2.to_csv(os.path.join(pathsave, 'value_per_frame.csv'),index=False)


def create_all_still(pathsave,img_o,maskDraw,size,eps,thresh_top,sigma,small,angleA,overlap,name_cell):
    create_output_dirs(pathsave)

    N,P = (img_o.shape)
    imgP=np.zeros((N+2,P+2))
    imgP = np.pad(img_o, 1, 'constant')

    posL, graphTagg, imF = process_individual_image(imgP, pathsave, size, eps, thresh_top, sigma, small, angleA, overlap)


    pd_fil_info = utilsF.filament_info(imgP, graphTagg, posL, pathsave,imF,maskDraw)
    pd_fil_info = pd.read_csv(os.path.join(pathsave, 'traced_filaments_info.csv'))

    mean_len = np.mean(pd_fil_info['filament length'])
    list_len = np.sort(pd_fil_info['filament length'])
    plt.figure()
    plt.scatter( np.arange(0,len(list_len)),list_len)


    mean_angle,var_val = utilsF.circ_stat(pd_fil_info,pathsave)
    print('mean angle: ', mean_angle, 'circ var: ', var_val, 'mean length: ', mean_len)
