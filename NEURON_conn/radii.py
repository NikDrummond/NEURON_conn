# Functions for extracting radii from neuron skeletons

import skeletor as sk
import trimesh as tm
import numpy as np
import networkx as nx
import navis
import ncollpyde
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.spatial.distance import pdist


from shapely.geometry import LineString as shLs
from shapely.geometry import Point as shPt

import itertools

from operator import itemgetter

from scipy import stats


import numbers
import warnings

import pandas as pd
import scipy.spatial

import warnings

import plotly.graph_objects as go


from tqdm.auto import tqdm 



def center_fit(root_coord,end_coord,origin,mesh,method = 'mean', rays = 100,dist = 100000000):
    """ Given the root and end coordinates of a line, and an origin point, which is a point on that line.
    return a new central point, the sum of Squared error of residuals for elipses fit, the radius (defined as the
    mean of the two cardinal axis of the elipses), the squash of the elipses, defined as the difference between the 
    two radi of the elipses.
    
    methods - mean, median, ellipses
    
    """
    
    # get the direction vector of start and end point and normalise
    norm_dir = (end_coord-root_coord)
    norm_dir = norm_dir/np.linalg.norm(norm_dir)
    
    if method == 'mean':
        
        directions = ray_directions(norm_dir,rays)
        # get intersections of rays with mesh
        intersections = ray_intersections(mesh,directions,origin,dist)
        
        if len(intersections) <= rays * 0.9:
            # return nan
            output = pd.DataFrame([np.nan,np.nan,np.nan]).T 
            output.columns = ['x','y','z']
            output['radius'] = np.nan
        else:
            new = origin
            output = pd.DataFrame(new).T 
            output.columns = ['x','y','z']
            output['radius'] = np.mean(pdist(intersections))/2
            
    elif method == 'median':
        
        directions = ray_directions(norm_dir,rays)
        # get intersections of rays with mesh
        intersections = ray_intersections(mesh,directions,origin,dist)
        
        if len(intersections) <= 450:
            # return nan
            output = pd.DataFrame([np.nan,np.nan,np.nan]).T 
            output.columns = ['x','y','z']
            output['radius'] = np.nan
        else:
            new = origin
            output = pd.DataFrame(new).T 
            output.columns = ['x','y','z']
            output['radius'] = np.median(pdist(intersections))/2
            
    elif method == 'ellipses':\
        
            # keep a counter of attempts
        conv_count = 0
        converge = False
        while converge == False:

            directions = ray_directions(norm_dir,rays)
            # get intersections of rays with mesh
            intersections = ray_intersections(mesh,directions,origin,dist)



            if len(intersections) <= 450:
                conv_count += 1

            else:

                #generate 2D axis
                x_axis,y_axis = TwoD_axis(norm_dir)

                # Project intersections and current origin point onto 2D space 
                TwoD_points = np.array([(np.dot(intersections[i],x_axis), np.dot(intersections[i],y_axis)) 
                                        for i in range(len(intersections))])
                cent = np.array([np.dot(origin,x_axis), np.dot(origin,y_axis)])

                # fit elipsemodel            
                ell = EllipseModel()
                ell.estimate(TwoD_points)
                converge = ell.estimate(TwoD_points)

                # if it converges, generate output
                if converge == True:
                    # get elipses params
                    xc, yc, a, b, theta = ell.params
                    # caluclate residuals of fit
                    residuals = ell.residuals(TwoD_points)
                    # calculate sum of least squares of fit
                    SofS_error = sum(list(map(lambda n: n**2,residuals)))
                    # get the radius as the mean of the two cardinal axis
                    radius = np.mean([a,b])
                    # get the difference between the axis as a measure of 'squash'
                    squash = abs(a-b)
                    # get the new central point in 2D
                    center = np.array([xc, yc])
                    # get the absolute distance betweent the new nd old central point
                    dist = abs(np.sqrt( (center[0] - cent[0])**2 +  (center[1] - cent[1])**2))
                    # Find which ray the new central point is closest to, and get the direction of the ray.
                    # Use this and the distance to get the new central possition
                    distances = [dist_to_line(cent,TwoD_points[i],center) for i in range(len(TwoD_points))]
                    new = origin + (dist * directions[distances.index(min(distances))])

                    output = pd.DataFrame(new).T 
                    output.columns = ['x','y','z']
                    output['radius'] = radius
                else:
                    conv_count +=1

            if conv_count >= 10:
                converge = True
                output = pd.DataFrame([np.nan,np.nan,np.nan]).T 
                output.columns = ['x','y','z']
                output['radius'] = np.nan

    return output



def getEquiPts(coords,parts):
    mid_points = np.array(list(zip(*[np.linspace(coords.loc[0,i], coords.loc[1,i], 
                                              parts+1, endpoint = False) for i in ['x','y','z']])))
    mid_points = np.delete(mid_points,0,0)

    mid_points = pd.DataFrame(mid_points)
    mid_points.columns = ['x','y','z']
    return mid_points

def ray_directions(norm_dir,rays):
    """ given a 3D origin point returns 'rays' number of directions randomly orthogonal to 
    the origin
    """
    direction = np.zeros((rays,3))
    for i in range(rays) :
        # create random vector
        rand = np.array(np.random.standard_normal(3))
        rand = rand/np.linalg.norm(rand)

        # get cross product of original line and random vector ( gives the orthogonal direction)
        d = np.cross(norm_dir,rand)
        d =  d/np.linalg.norm(d)
        
        # modify row in array
        direction[i] = d
    return(direction)

def TwoD_axis(norm_dir):
    """ define a set of x/y axis based on the normal of our line,
    In order to project points onto two dimensions
    """
    
    # define x axis
    x = np.array([1,0,0])
    # project this point onto the plane defined by the direction vector
    x = x - np.dot(x,norm_dir) * norm_dir
    # make a unit vector
    x /= np.sqrt((x**2).sum())
    # define y axis - vector orthogonal to the normal of the plane and x
    y = np.cross(norm_dir,x)
    return(x,y)

def ray_intersections(mesh, directions, origin, length):
    """ When provided with a trimesh mesh, a set of directions to cast your rays, 
     and orgin point to cats them from, and a length of the ray, returns the intersection coordinates with the mesh"""
    
    # Get end points of rays
    ray_ends = np.array([origin + (length * directions[i]) for i in range(len(directions))])

    # Initialize ncollpyde Volume
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)

    # get coordinates of first intersection with mesh face
    intersections = coll.intersections(np.array([origin,]*len(directions)),ray_ends)[1]
    
    return intersections
    
def dist_to_line(origin,boundary,point):
    """ Gives the distance of a point to a line in 2D. inputs should be 1x2 arrays of x,y coords"""
    l = shLs([ (origin[0],origin[1]), (boundary[0],boundary[1])])
    p = shPt(point[0],point[1])
    return p.distance(l)

def LoS_matrix(vol, swc):
    """ 
    Creates a line of sight within the mesh binary matrix for all nodes 
    to all other nodes within a neurons
    
    input = ncollpyde volume, swc table
    
    output: matrix of true/False values and dict of node - index mappings
    """
    # get node ids - not really needed
    ids = list(swc['node_id'])
    # create list of all by all for source and target corrdinates
    source = np.array(list((itertools.chain.from_iterable(itertools.repeat(swc.loc[swc['node_id']==x,['node_id','x','y','z']].values, len(ids)) for x in ids))))[:,0,:]
    target = np.tile(swc[['node_id','x','y','z']].values,(len(ids),1))
    
    # index of rays without  line of sight
    LoS_ix = vol.intersections(source[:,1:4], target[:,1:4])[0]
    
    #initialise Line of Sight matrix
    LoS_mat = np.full((len(swc['node_id']),len(swc['node_id'])),False)
    
    # get the index in source/target of pairs with line of sight
    ind = list(set(range(len(source))) - set(LoS_ix))
    
    # create a disctionary of node ids to index in node table
    node_indxMap = {swc['node_id'][i]:i for i in swc.index}
    
    # get x and y indicies of pairs with line of sight
    x = [node_indxMap[i] for i in source[ind,0]]
    y = [node_indxMap[i] for i in target[ind,0]]
    # update these 
    LoS_mat[x,y] = True
    LoS_mat[y,x] = True
    
    LoS_mat = pd.DataFrame(LoS_mat)
    LoS_mat.index = list(node_indxMap.keys())
    LoS_mat.columns = list(node_indxMap.keys())
    
    return LoS_mat

