# functions for manipulating neuron skeleton geometry in some way

from .skeletons import *
import numpy as np
import pandas as pd
import networkx as nx
import navis
import skeletor as sk






def shrink_ends(swc, shrink):
    
    # using list comprehensions as vectorisation was broadcasting weirdly and I clearly don't get it 
    
    
    ####
    
    # TODO - shift root in a touch as well! 
    
    ####
    
    ends = list(set(swc['node_id']) - set(swc['parent_id']))
    parents = swc.loc[swc['node_id'].isin(ends)]['parent_id'].values

    end_coords = swc.loc[swc['node_id'].isin(ends)][['x','y','z']].values
    parent_coords = pd.concat([swc.loc[swc['node_id'] == swc.loc[swc['node_id']==i]['parent_id'].values[0]][['x','y','z']] 
                               for i in ends]).values
    
    dist = [np.linalg.norm(end_coords[i]-parent_coords[i]) for i in range(len(end_coords))]

    direction = [end_coords[i] - parent_coords[i] for i in range(len(end_coords))]
    direction = [direction[i]/np.linalg.norm(direction[i]) for i in range(len(end_coords))]
    
    new = [parent_coords[i] + ((dist[i]*shrink) * direction[i]) for i in range(len(end_coords))]
    
    swc.loc[swc['node_id'].isin(ends),['x','y','z']] = new
    
    return swc


def path_dictionary(path, LoS_mat, r = -1):
    path_dic = {}
    i = path[0]
    j = i
    
    while (i != r):
        if (j == r):
            j = path[(path.index(j))-1]
            path_dic[i] = j
            break
        if LoS_mat.loc[i,j] == True:
            if j != r:
                j = path[(path.index(j)) + 1]
        elif LoS_mat.loc[i,j] == False:
            if i == path[(path.index(j))-1]:
                i = path[(path.index(j))+1]
                j = i
            else:        
                j = path[(path.index(j))-1]
                path_dic[i] = j
                i = j
    return path_dic

def single_skeleton(swc, mesh):
    """ returns an slightly cleaned swc table returned from skeletonisation as a single connected neuron graph
    
    """
    nodes = swc.set_index('node_id', inplace=False)
    # create an empty graph:
    G = nx.Graph()
    G.add_nodes_from(list(swc['node_id']))
    all_coords = dict(zip(swc['node_id'],swc[['x','y','z']].values))

    # add edges

    edges = swc[swc.parent_id >=0][['node_id', 'parent_id']].values
    weights = np.sqrt(np.sum((nodes.loc[edges[:, 0], ['x', 'y', 'z']].values.astype(float)
                                  - nodes.loc[edges[:, 1], ['x', 'y', 'z']].values.astype(float)) ** 2, axis=1))

    edge_dict = np.array([{'weight': w} for w in weights])
    # Add weights to dictionary
    edges = np.append(edges, edge_dict.reshape(len(edges), 1), axis=1)

    G.add_edges_from(edges)
    
    # remove any disconected component with less than 0.0001% of the total nodes
    trim = 0.001
    for component in list(nx.connected_components(G)):
        if len(component)<len(G.nodes) * trim:
            for node in component:
                G.remove_node(node)
                
    # creat a list of sub graphs
    N_fragments = [G.subgraph(x) for x in list(nx.connected_components(G))]
    # for each sub graph, turn into neuron/neuronlist
    N_all = navis.NeuronList([])
    for frag in N_fragments:
        # turn into directed graph
        new_edges = []
        g = frag.copy()
        # select the first node as the root
        r = list(g.nodes)[0]
        # create edge list
        this_lop = nx.predecessor(g, r)
        new_edges += [(k, v[0]) for k, v in this_lop.items() if v]
        # initilise directed graph
        g = nx.DiGraph()
        g.add_edges_from(new_edges)  

        # turn into SWC
        swc = sk.skeletonizers.make_swc(g,mesh)

        # update the coordinate values in swc, as they are wrong when pulled from the mesh
        # get coords of nodes in graph
        for node in list(swc['node_id']):
            swc.loc[swc.node_id==node,'x'] = all_coords[node][0]
            swc.loc[swc.node_id==node,'y'] = all_coords[node][1]
            swc.loc[swc.node_id==node,'z'] = all_coords[node][2]

        # Iteratively remove line of sight twigs
        swc = rem_line_of_sight(swc,mesh)
        
        
        # using clean here seemed to produce nodes with identical coordinates
        # Until this is fixed in some way, leave
        # swc = sk.clean(swc,mesh)
        
        
        # FUNCTION IN HERE TO REMOVE X,Y,Z COORD DUPLICATES



        # add to neuronlistncollpyde
        N_all += navis.TreeNeuron(swc)

    N = navis.stitch_neurons(N_all,method = 'LEAFS',master = 'LARGEST')
    # navis.downsample_neuron(N, downsampling_factor = 2, inplace = True)
    
    return N