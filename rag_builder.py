from skimage.future import graph
import numpy as np
import math

def assign_weights(g, mode):
    '''
    Assigns weights to the edges of the underlying graph. These depend on
    the mode of weights.
    '''

    if mode == 'similarity':
        sigma=255
        for x, y, d in g.edges(data=True):
            diff = g.nodes[x]['mean color'] - g.nodes[y]['mean color']
            diff = np.linalg.norm(diff)
            d['weight'] = math.e ** (-(diff ** 2) / sigma)

    elif mode == 'laplacian':
        Laplacian = np.load('Laplacian.npy')
        for x, y, d in g.edges(data=True):
            d['weight'] = np.absolute(Laplacian[(x,y)])

    elif mode == 'experiment':
        # Replace "1" with weight of choice.
        for x, y, d in g.edges(data=True):
            d['weight'] = 1
    else:
        raise ValueError("The mode '%s' is not recognised" % mode)

def build_rag(labels, weight_mode):
    '''
    Constructs region adjacency graph for segmented image.
    '''

    print("Constructing graph...")
    g = graph.RAG(labels, connectivity=2)

    for n in g:
        g.nodes[n].update({'labels': [n]})

    mean_colors = np.load('regions.npy')

    for n in g:
        g.nodes[n]['mean color'] = mean_colors[n]

    assign_weights(g, weight_mode)

    return g
