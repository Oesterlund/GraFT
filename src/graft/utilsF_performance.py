import math

from line_profiler import profile
from scipy import ndimage, sparse
import skimage
from skimage.measure import label
import numpy as np
import scipy as sp


def node_condense_original(imageFiltered,imageSkeleton,kernel):
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


def node_condense_01(imageFiltered,imageSkeleton,kernel):
    '''
    modified version of utilsF.node_condense with the following differences:
    
    - calculate `small` only once
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


def node_condense_02(imageFiltered,imageSkeleton,kernel):
    '''
    modified version of utilsF.node_condense with the following differences:
    
    - calculate `small` only once
    - calculate `location` only once inside if((np.sum((small>1)*1)>2)) block
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


def node_condense_03(imageFiltered,imageSkeleton,kernel):
    '''
    modified version of utilsF.node_condense with the following differences:
    
    - calculate `small` only once
    - calculate `location` only once inside if((np.sum((small>1)*1)>2)) block
    - precalculate `connectivity`
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
            if np.sum((small > 1) * 1) > 2:
                location=np.argwhere(small > 1)
                connectivity = {}
                for coord in location:
                    coord_tuple = (coord[0], coord[1])
                    local_sum = np.sum((imgSL[l-half+coord[0]-1:l-half+coord[0]+2, k-half+coord[1]-1:k-half+coord[1]+2] > 0) * 1)
                    connectivity[coord_tuple] = local_sum

                for z, coord in enumerate(location):
                    if connectivity[(coord[0], coord[1])] == 2:
                        imgSL[l-half+coord[0], k-half+coord[1]] = 1

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


def node_condense_04(imageFiltered, imageSkeleton, kernel):
    '''
    modified version of utilsF.node_condense with the following differences:
    
    - calculate `small` only once
    - calculate `location` only once inside if((np.sum((small>1)*1)>2)) block
    - precalculate `connectivity`
    - GPT-4o cleanup
    '''
    imageLabeled, labels = ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    # need to have rolling window to condense nodes together
    imgSL = imageLabeled + imageSkeleton

    half = int(len(kernel) / 2)
    M, N = imgSL.shape

    for l in range(half, M - half):
        for k in range(half, N - half):
            small = imgSL[l - half:l + half, k - half:k + half]

            # get all pixel location for vals higher than 1
            if np.sum((small > 1) * 1) > 2:
                location = np.argwhere(small > 1)
                connectivity = {}
                
                for coord in location:
                    coord_tuple = (coord[0], coord[1])
                    local_sum = np.sum((imgSL[l - half + coord[0] - 1:l - half + coord[0] + 2, k - half + coord[1] - 1:k - half + coord[1] + 2] > 0) * 1)
                    connectivity[coord_tuple] = local_sum

                for coord in location:
                    if connectivity[(coord[0], coord[1])] == 2:
                        imgSL[l - half + coord[0], k - half + coord[1]] = 1

                location = np.argwhere(small > 1)
                for i, coord in enumerate(location):
                    if i != np.floor(len(location) / 2):
                        imgSL[l - half + coord[0], k - half + coord[1]] = 1

            # there are two different points close to each other
            elif np.sum((small > 1) * 1) == 2:
                location = np.argwhere(small > 1)

                # test that both points are not endnodes
                coord1 = location[0]
                coord2 = location[1]
                
                node1conn = np.sum((imgSL[l - half + coord1[0] - 1:l - half + coord1[0] + 2, k - half + coord1[1] - 1:k - half + coord1[1] + 2] > 0) * 1)
                node2conn = np.sum((imgSL[l - half + coord2[0] - 1:l - half + coord2[0] + 2, k - half + coord2[1] - 1:k - half + coord2[1] + 2] > 0) * 1)
                
                if node1conn > 2 and node2conn > 2:
                    # not end nodes
                    y_min, y_max = min(l - half + coord1[0], l - half + coord2[0]), max(l - half + coord1[0], l - half + coord2[0])
                    x_min, x_max = min(k - half + coord1[1], k - half + coord2[1]), max(k - half + coord1[1], k - half + coord2[1])
                    zoom = (imgSL[y_min:y_max + 1, x_min:x_max + 1] > 0) * 1
                    conV = skimage.measure.label(zoom, connectivity=2)
                    
                    if np.max(conV) == 1:
                        com = ndimage.center_of_mass(conV)
                        if coord1[1] < coord2[1]:
                            coordinateC = (math.ceil(com[0]) + coord1[0], math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], math.floor(com[1]) + coord1[1])
                        else:
                            coordinateC = (math.ceil(com[0]) + coord1[0], -math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], -math.floor(com[1]) + coord1[1])
                        
                        # test if the coordinate has value different from 0
                        if imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] = val
                        elif imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] = val
                    elif np.max(conV) == 2:
                        # need to check that the two nodes are connected
                        arr, num = skimage.measure.label((small > 0) * 1, return_num=True, connectivity=2)
                        if num == 1:
                            # remove end node
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                else:
                    # need to check that the two nodes are connected
                    arr, num = skimage.measure.label((small > 0) * 1, return_num=True, connectivity=2)
                    if num == 1:
                        # remove end node
                        if node1conn == 2:
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                        if node2conn == 2:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                        
    return imgSL - imageSkeleton


def node_condense_05(imageFiltered, imageSkeleton, kernel):
    '''
    modified version of utilsF.node_condense with the following differences:
    
    - calculate `small` only once
    - calculate `location` only once inside if((np.sum((small>1)*1)>2)) block
    - precalculate `connectivity`
    - GPT-4o cleanup    
    - calculate only once: small_sum = np.sum((small > 1) * 1)
    
    Surprisingly, `small_sum` was the biggest performance increase so far (reducing runtime from 15s to 8s)!
    '''
    imageLabeled, labels = ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    imgSL = imageLabeled + imageSkeleton

    half = int(len(kernel) / 2)
    M, N = imgSL.shape

    for l in range(half, M - half):
        for k in range(half, N - half):
            small = imgSL[l - half:l + half, k - half:k + half]
            small_sum = np.sum((small > 1) * 1)

            if small_sum > 2:
                location = np.argwhere(small > 1)
                connectivity = {}
                
                for coord in location:
                    coord_tuple = (coord[0], coord[1])
                    local_sum = np.sum((imgSL[l - half + coord[0] - 1:l - half + coord[0] + 2, k - half + coord[1] - 1:k - half + coord[1] + 2] > 0) * 1)
                    connectivity[coord_tuple] = local_sum

                for coord in location:
                    if connectivity[(coord[0], coord[1])] == 2:
                        imgSL[l - half + coord[0], k - half + coord[1]] = 1

                location = np.argwhere(small > 1)
                for i, coord in enumerate(location):
                    if i != np.floor(len(location) / 2):
                        imgSL[l - half + coord[0], k - half + coord[1]] = 1

            elif small_sum == 2:
                location = np.argwhere(small > 1)
                coord1, coord2 = location[0], location[1]
                
                node1conn = np.sum((imgSL[l - half + coord1[0] - 1:l - half + coord1[0] + 2, k - half + coord1[1] - 1:k - half + coord1[1] + 2] > 0) * 1)
                node2conn = np.sum((imgSL[l - half + coord2[0] - 1:l - half + coord2[0] + 2, k - half + coord2[1] - 1:k - half + coord2[1] + 2] > 0) * 1)
                
                if node1conn > 2 and node2conn > 2:
                    y_min, y_max = min(l - half + coord1[0], l - half + coord2[0]), max(l - half + coord1[0], l - half + coord2[0])
                    x_min, x_max = min(k - half + coord1[1], k - half + coord2[1]), max(k - half + coord1[1], k - half + coord2[1])
                    zoom = (imgSL[y_min:y_max + 1, x_min:x_max + 1] > 0) * 1
                    conV = skimage.measure.label(zoom, connectivity=2)
                    
                    if np.max(conV) == 1:
                        com = ndimage.center_of_mass(conV)
                        if coord1[1] < coord2[1]:
                            coordinateC = (math.ceil(com[0]) + coord1[0], math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], math.floor(com[1]) + coord1[1])
                        else:
                            coordinateC = (math.ceil(com[0]) + coord1[0], -math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], -math.floor(com[1]) + coord1[1])
                        
                        if imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] = val
                        elif imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] = val
                    elif np.max(conV) == 2:
                        arr, num = skimage.measure.label((small > 0) * 1, return_num=True, connectivity=2)
                        if num == 1:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                else:
                    arr, num = skimage.measure.label((small > 0) * 1, return_num=True, connectivity=2)
                    if num == 1:
                        if node1conn == 2:
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                        if node2conn == 2:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                        
    return imgSL - imageSkeleton


def node_condense_06(imageFiltered, imageSkeleton, kernel):
    '''
    GPT-4o suggested this version that uses sparse matrixes, but it's 10x slower
    than node_condense_05!
    '''
    imageLabeled, labels = ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    imgSL = sparse.csr_matrix(imageLabeled + imageSkeleton)

    half = int(len(kernel) / 2)
    M, N = imgSL.shape

    for l in range(half, M - half):
        for k in range(half, N - half):
            small = imgSL[l - half:l + half, k - half:k + half].toarray()
            small_sum = np.sum(small > 1)

            if small_sum > 2:
                location = np.argwhere(small > 1)
                connectivity = {}

                for coord in location:
                    coord_tuple = (coord[0], coord[1])
                    local_sum = np.sum(imgSL[l - half + coord[0] - 1:l - half + coord[0] + 2, k - half + coord[1] - 1:k - half + coord[1] + 2].toarray() > 0)
                    connectivity[coord_tuple] = local_sum

                for coord in location:
                    if connectivity[(coord[0], coord[1])] == 2:
                        imgSL[l - half + coord[0], k - half + coord[1]] = 1

                location = np.argwhere(small > 1)
                for i, coord in enumerate(location):
                    if i != np.floor(len(location) / 2):
                        imgSL[l - half + coord[0], k - half + coord[1]] = 1

            elif small_sum == 2:
                location = np.argwhere(small > 1)
                coord1, coord2 = location[0], location[1]

                node1conn = np.sum(imgSL[l - half + coord1[0] - 1:l - half + coord1[0] + 2, k - half + coord1[1] - 1:k - half + coord1[1] + 2].toarray() > 0)
                node2conn = np.sum(imgSL[l - half + coord2[0] - 1:l - half + coord2[0] + 2, k - half + coord2[1] - 1:k - half + coord2[1] + 2].toarray() > 0)

                if node1conn > 2 and node2conn > 2:
                    y_min, y_max = min(l - half + coord1[0], l - half + coord2[0]), max(l - half + coord1[0], l - half + coord2[0])
                    x_min, x_max = min(k - half + coord1[1], k - half + coord2[1]), max(k - half + coord1[1], k - half + coord2[1])
                    zoom = imgSL[y_min:y_max + 1, x_min:x_max + 1].toarray() > 0
                    conV = skimage.measure.label(zoom, connectivity=2)

                    if np.max(conV) == 1:
                        com = ndimage.center_of_mass(conV)
                        if coord1[1] < coord2[1]:
                            coordinateC = (math.ceil(com[0]) + coord1[0], math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], math.floor(com[1]) + coord1[1])
                        else:
                            coordinateC = (math.ceil(com[0]) + coord1[0], -math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], -math.floor(com[1]) + coord1[1])

                        if imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] = val
                        elif imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] = val
                    elif np.max(conV) == 2:
                        arr, num = skimage.measure.label(small > 0, return_num=True, connectivity=2)
                        if num == 1:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                else:
                    arr, num = skimage.measure.label(small > 0, return_num=True, connectivity=2)
                    if num == 1:
                        if node1conn == 2:
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                        if node2conn == 2:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1

    return imgSL.toarray() - imageSkeleton


def node_condense_07(imageFiltered, imageSkeleton, kernel):
    '''
    modified version of utilsF.node_condense with the following differences:
    
    - calculate `small` only once
    - calculate `location` only once inside if((np.sum((small>1)*1)>2)) block
    - precalculate `connectivity`
    - GPT-4o cleanup    
    - calculate only once: small_sum = np.sum((small > 1) * 1)
    - vectorized some operations

    Surprisingly, `small_sum` was the biggest performance increase so far (reducing runtime from 15s to 8s)!
    Vectorization saves 2s, the remaining changes had no measurable impact.
    '''
    imageLabeled, labels = ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    imgSL = imageLabeled + imageSkeleton

    half = len(kernel) // 2
    M, N = imgSL.shape

    for l in range(half, M - half):
        for k in range(half, N - half):
            small = imgSL[l - half:l + half, k - half:k + half]
            small_sum = np.sum(small > 1)

            if small_sum > 2:
                location = np.argwhere(small > 1)
                local_sums = np.array([np.sum(imgSL[l - half + coord[0] - 1:l - half + coord[0] + 2, k - half + coord[1] - 1:k - half + coord[1] + 2] > 0) for coord in location])
                connectivity = dict(zip(map(tuple, location), local_sums))

                to_update = [coord for coord in location if connectivity[tuple(coord)] == 2]
                to_update = np.array(to_update)
                if to_update.size > 0:
                    imgSL[(l - half + to_update[:, 0], k - half + to_update[:, 1])] = 1

                location = np.argwhere(small > 1)
                center_index = np.floor(len(location) / 2).astype(int)
                if len(location) > 0:
                    location = np.delete(location, center_index, axis=0)
                    imgSL[(l - half + location[:, 0], k - half + location[:, 1])] = 1

            elif small_sum == 2:
                location = np.argwhere(small > 1)
                coord1, coord2 = location[0], location[1]

                node1conn = np.sum(imgSL[l - half + coord1[0] - 1:l - half + coord1[0] + 2, k - half + coord1[1] - 1:k - half + coord1[1] + 2] > 0)
                node2conn = np.sum(imgSL[l - half + coord2[0] - 1:l - half + coord2[0] + 2, k - half + coord2[1] - 1:k - half + coord2[1] + 2] > 0)

                if node1conn > 2 and node2conn > 2:
                    y_min, y_max = min(l - half + coord1[0], l - half + coord2[0]), max(l - half + coord1[0], l - half + coord2[0])
                    x_min, x_max = min(k - half + coord1[1], k - half + coord2[1]), max(k - half + coord1[1], k - half + coord2[1])
                    zoom = imgSL[y_min:y_max + 1, x_min:x_max + 1] > 0
                    conV = skimage.measure.label(zoom, connectivity=2)

                    if np.max(conV) == 1:
                        com = ndimage.center_of_mass(conV)
                        if coord1[1] < coord2[1]:
                            coordinateC = (math.ceil(com[0]) + coord1[0], math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], math.floor(com[1]) + coord1[1])
                        else:
                            coordinateC = (math.ceil(com[0]) + coord1[0], -math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], -math.floor(com[1]) + coord1[1])

                        if imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] = val
                        elif imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] = val
                    elif np.max(conV) == 2:
                        arr, num = skimage.measure.label(small > 0, return_num=True, connectivity=2)
                        if num == 1:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                else:
                    arr, num = skimage.measure.label(small > 0, return_num=True, connectivity=2)
                    if num == 1:
                        if node1conn == 2:
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                        if node2conn == 2:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1

    return imgSL - imageSkeleton


def node_condense_10(imageFiltered, imageSkeleton, kernel):
    """
    diff from node_condense_07: vectorized local_sums
    """
    imageLabeled, labels = ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    imgSL = imageLabeled + imageSkeleton

    half = len(kernel) // 2
    M, N = imgSL.shape

    for l in range(half, M - half):
        for k in range(half, N - half):
            small = imgSL[l - half:l + half, k - half:k + half]
            small_sum = np.sum(small > 1)

            if small_sum > 2:
                location = np.argwhere(small > 1)
                local_sums = np.zeros(len(location))

                for idx, (i, j) in enumerate(location):
                    local_patch = imgSL[l - half + i - 1:l - half + i + 2, k - half + j - 1:k - half + j + 2]
                    local_sums[idx] = np.sum(local_patch > 0)

                connectivity = dict(zip(map(tuple, location), local_sums))

                to_update = [coord for coord in location if connectivity[tuple(coord)] == 2]
                to_update = np.array(to_update)
                if to_update.size > 0:
                    imgSL[(l - half + to_update[:, 0], k - half + to_update[:, 1])] = 1

                location = np.argwhere(small > 1)
                center_index = np.floor(len(location) / 2).astype(int)
                if len(location) > 0:
                    location = np.delete(location, center_index, axis=0)
                    imgSL[(l - half + location[:, 0], k - half + location[:, 1])] = 1

            elif small_sum == 2:
                location = np.argwhere(small > 1)
                coord1, coord2 = location[0], location[1]

                node1conn = np.sum(imgSL[l - half + coord1[0] - 1:l - half + coord1[0] + 2, k - half + coord1[1] - 1:k - half + coord1[1] + 2] > 0)
                node2conn = np.sum(imgSL[l - half + coord2[0] - 1:l - half + coord2[0] + 2, k - half + coord2[1] - 1:k - half + coord2[1] + 2] > 0)

                if node1conn > 2 and node2conn > 2:
                    y_min, y_max = min(l - half + coord1[0], l - half + coord2[0]), max(l - half + coord1[0], l - half + coord2[0])
                    x_min, x_max = min(k - half + coord1[1], k - half + coord2[1]), max(k - half + coord1[1], k - half + coord2[1])
                    zoom = imgSL[y_min:y_max + 1, x_min:x_max + 1] > 0
                    conV = skimage.measure.label(zoom, connectivity=2)

                    if np.max(conV) == 1:
                        com = ndimage.center_of_mass(conV)
                        if coord1[1] < coord2[1]:
                            coordinateC = (math.ceil(com[0]) + coord1[0], math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], math.floor(com[1]) + coord1[1])
                        else:
                            coordinateC = (math.ceil(com[0]) + coord1[0], -math.ceil(com[1]) + coord1[1])
                            coordinateF = (math.floor(com[0]) + coord1[0], -math.floor(com[1]) + coord1[1])

                        if imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] = val
                        elif imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] > 0:
                            val = imgSL[l - half + coord1[0], k - half + coord1[1]]
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                            imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] = val
                    elif np.max(conV) == 2:
                        arr, num = skimage.measure.label(small > 0, return_num=True, connectivity=2)
                        if num == 1:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
                else:
                    arr, num = skimage.measure.label(small > 0, return_num=True, connectivity=2)
                    if num == 1:
                        if node1conn == 2:
                            imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
                        if node2conn == 2:
                            imgSL[l - half + coord2[0], k - half + coord2[1]] = 1

    return imgSL - imageSkeleton


def label_images(imageFiltered, imageSkeleton):
    imageLabeled, labels = ndimage.label(imageFiltered, structure=np.ones((3, 3)))
    return imageLabeled + imageSkeleton


def calculate_local_sums(small, imgSL, l, k, half):
    location = np.argwhere(small > 1)
    local_sums = np.zeros(len(location))

    for idx, (i, j) in enumerate(location):
        local_patch = imgSL[l - half + i - 1:l - half + i + 2, k - half + j - 1:k - half + j + 2]
        local_sums[idx] = np.sum(local_patch > 0)

    return location, local_sums


def update_imgSL(imgSL, l, k, half, location, local_sums):
    connectivity = dict(zip(map(tuple, location), local_sums))
    to_update = [coord for coord in location if connectivity[tuple(coord)] == 2]
    to_update = np.array(to_update)
    if to_update.size > 0:
        imgSL[(l - half + to_update[:, 0], k - half + to_update[:, 1])] = 1

    location = np.argwhere(imgSL[l - half:l + half, k - half:k + half] > 1)
    center_index = np.floor(len(location) / 2).astype(int)
    if len(location) > 0:
        location = np.delete(location, center_index, axis=0)
        imgSL[(l - half + location[:, 0], k - half + location[:, 1])] = 1


def handle_two_nodes(imgSL, l, k, half, location, small):
    coord1, coord2 = location[0], location[1]
    node1conn = np.sum(imgSL[l - half + coord1[0] - 1:l - half + coord1[0] + 2, k - half + coord1[1] - 1:k - half + coord1[1] + 2] > 0)
    node2conn = np.sum(imgSL[l - half + coord2[0] - 1:l - half + coord2[0] + 2, k - half + coord2[1] - 1:k - half + coord2[1] + 2] > 0)

    if node1conn > 2 and node2conn > 2:
        y_min, y_max = min(l - half + coord1[0], l - half + coord2[0]), max(l - half + coord1[0], l - half + coord2[0])
        x_min, x_max = min(k - half + coord1[1], k - half + coord2[1]), max(k - half + coord1[1], k - half + coord2[1])
        zoom = imgSL[y_min:y_max + 1, x_min:x_max + 1] > 0
        conV = label(zoom, connectivity=2)

        if np.max(conV) == 1:
            com = ndimage.center_of_mass(conV)
            coordinateC, coordinateF = calculate_coordinates(coord1, coord2, com)
            update_coordinates(imgSL, l, k, half, coord1, coord2, coordinateC, coordinateF)
        elif np.max(conV) == 2:
            arr, num = label(small > 0, return_num=True, connectivity=2)
            if num == 1:
                imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
    else:
        arr, num = label(small > 0, return_num=True, connectivity=2)
        if num == 1:
            if node1conn == 2:
                imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
            if node2conn == 2:
                imgSL[l - half + coord2[0], k - half + coord2[1]] = 1


def calculate_coordinates(coord1, coord2, com):
    if coord1[1] < coord2[1]:
        coordinateC = (math.ceil(com[0]) + coord1[0], math.ceil(com[1]) + coord1[1])
        coordinateF = (math.floor(com[0]) + coord1[0], math.floor(com[1]) + coord1[1])
    else:
        coordinateC = (math.ceil(com[0]) + coord1[0], -math.ceil(com[1]) + coord1[1])
        coordinateF = (math.floor(com[0]) + coord1[0], -math.floor(com[1]) + coord1[1])
    return coordinateC, coordinateF


def update_coordinates(imgSL, l, k, half, coord1, coord2, coordinateC, coordinateF):
    if imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] > 0:
        val = imgSL[l - half + coord1[0], k - half + coord1[1]]
        imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
        imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
        imgSL[l - half + coordinateC[0], k - half + coordinateC[1]] = val
    elif imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] > 0:
        val = imgSL[l - half + coord1[0], k - half + coord1[1]]
        imgSL[l - half + coord1[0], k - half + coord1[1]] = 1
        imgSL[l - half + coord2[0], k - half + coord2[1]] = 1
        imgSL[l - half + coordinateF[0], k - half + coordinateF[1]] = val


@profile
def node_condense_11(imageFiltered, imageSkeleton, kernel):
    """modularized version of node_condense_10"""
    imgSL = label_images(imageFiltered, imageSkeleton)
    half = len(kernel) // 2
    M, N = imgSL.shape

    for l in range(half, M - half):
        for k in range(half, N - half):
            small = imgSL[l - half:l + half, k - half:k + half]
            small_sum = np.sum(small > 1)

            if small_sum > 2:
                location, local_sums = calculate_local_sums(small, imgSL, l, k, half)
                update_imgSL(imgSL, l, k, half, location, local_sums)
            elif small_sum == 2:
                location = np.argwhere(small > 1)
                handle_two_nodes(imgSL, l, k, half, location, small)

    return imgSL - imageSkeleton
