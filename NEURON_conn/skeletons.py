# skeletonisation functions largely ported fom skeletor for skeletonising
# volumetric neuron meshes

import skeletor as sk 
import ncollpyde
import numpy as np
import scipy
import networkx as nx


def rem_line_of_sight(swc, mesh, max_dist = 'auto', copy = True):
    """ modified version of remove line of sight function in Skeletor, continues iteratively until no pairs of ends have line of sight to each other"""
    if copy is True:
        swc = swc.copy()

    # Add distance to parents
    swc['parent_dist'] = 0
    not_root = swc.parent_id >= 0
    co1 = swc.loc[not_root, ['x', 'y', 'z']].values
    co2 = swc.set_index('node_id').loc[swc.loc[not_root, 'parent_id'],
                                    ['x', 'y', 'z']].values
    # adds distance to parent column to swc
    swc.loc[not_root, 'parent_dist'] = np.sqrt(np.sum((co1 - co2)**2, axis=1))


    to_collapse = [0,0]
    count = 0
    to_remove = []
    while (len(to_collapse) != 0):
        count += 1
        # print([count, ' collapsing: ',len(to_collapse)])

        # Add distance to parents
        swc['parent_dist'] = 0
        not_root = swc.parent_id >= 0
        co1 = swc.loc[not_root, ['x', 'y', 'z']].values
        co2 = swc.set_index('node_id').loc[swc.loc[not_root, 'parent_id'],
                                        ['x', 'y', 'z']].values
        # adds distance to parent column to swc
        swc.loc[not_root, 'parent_dist'] = np.sqrt(np.sum((co1 - co2)**2, axis=1))


        # If max dist is 'auto', we will use the longest child->parent edge in the
        # skeleton as limit
        if max_dist == 'auto':
            max_dist = swc.parent_dist.max()

        # Initialize ncollpyde Volume
        coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)

        # Find twigs
        twigs = swc[~swc.node_id.isin(swc.parent_id)]

        # Remove twigs that aren't inside the volume
        twigs = twigs[coll.contains(twigs[['x', 'y', 'z']].values)]


            
        # Generate rays between all pairs of twigs
        twigs_co = twigs[['x', 'y', 'z']].values
        sources = np.repeat(twigs_co, twigs.shape[0], axis=0)
        targets = np.tile(twigs_co, (twigs.shape[0], 1))

        # Keep track of indices
        pairs = np.stack((np.repeat(twigs.node_id, twigs.shape[0]),
                            np.tile(twigs.node_id, twigs.shape[0]))).T

        # If max distance, drop pairs that are too far appart
        if max_dist:
            d = scipy.spatial.distance.pdist(twigs_co)
            d = scipy.spatial.distance.squareform(d)
            is_close = d.flatten() <= max_dist
            pairs = pairs[is_close]
            sources = sources[is_close]
            targets = targets[is_close]

        # Drop self rays
        not_self = pairs[:, 0] != pairs[:, 1]
        sources, targets = sources[not_self], targets[not_self]
        pairs = pairs[not_self]

        # Get intersections: `ix` points to index of line segment; `loc` is the
        #  x/y/z coordinate of the intersection and `is_backface` is True if
        # intersection happened at the inside of a mesh
        ix, loc, is_backface = coll.intersections(sources, targets)

        # Find pairs of twigs with no intersection - i.e. with line of sight
        los = ~np.isin(np.arange(pairs.shape[0]), ix)

        # To collapse: have line of sight
        to_collapse = pairs[los]

        # Group into cluster we need to collapse
        G = nx.Graph()
        G.add_edges_from(to_collapse)
        clusters = nx.connected_components(G)


        # When collapsing the clusters, we need to decide which should be the
        # winning twig. For this we will use the twig lengths. In theory we ought to
        # be more fancy and ask for the distance to the root but that's more
        # expensive and it's unclear if it'll work any better.
        seg_lengths = twigs.set_index('node_id').parent_dist.to_dict()

        seen = set()
        for nodes in clusters:
            # We have to be careful not to generate chains here, e.g. A sees B,
            # B sees C, C sees D, etc. To prevent this, we will break up these
            # clusters into cliques and collapse them by order of size of the
            # cliques
            for cl in nx.find_cliques(nx.subgraph(G, nodes)):
                # Turn into set
                cl = set(cl)
                # Drop any node that has already been visited
                cl = cl - seen
                # Skip if less than 2 nodes left in clique
                if len(cl) < 2:
                    continue
                # Add nodes to list of visited nodes
                seen = seen | cl
                # Sort by segment lenghts to find the loosing nodes
                loosers = sorted(cl, key=lambda x: seg_lengths[x])[:-1]
                to_remove += loosers

        
        # Drop the tips we flagged for removal and the new column we added
        swc = swc[~swc.node_id.isin(to_remove)].drop('parent_dist', axis=1)

    return swc