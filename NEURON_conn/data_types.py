# converting data types (move to neurosetta?)

import networkx as nx
import numpy as np
import pandas as pd

def swc_to_graph(G,swc, directed = True):
    """ Convert either an swc table to a graph (undirected, or directed) or an undirected graph to a directed graph
    
    Parameters
    ----------
    
    G            pd.DataFrame | networkx.Graph
    
    directed     Bool
    
    Returns
    -------
    
    nx.Graph | nx.Digraph
    
    """
    
#    if(isinstance(G,nx.DiGraph)):
#        raise TypeError(f'Function has been provided a directed graph, so not needed')
        
#    if not (isinstance(G,pd.DataFrame)) or (isinstance(G,networkx.classes.graph.Graph)) :
#        raise TypeError(f'Expected Data Frame or undirected graph, got "{type(G)}"')    
    
    if(isinstance(G,pd.DataFrame)):

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

    if directed == True:

        # turn into directed graph
        new_edges = []
        # select the first node as the root
        r = list(G.nodes)[0]
        # create edge list
        this_lop = nx.predecessor(G, r)
        new_edges += [(k, v[0]) for k, v in this_lop.items() if v]
        # initilise directed graph
        g = nx.DiGraph()
        g.add_edges_from(new_edges)
        G = g
        
    return G