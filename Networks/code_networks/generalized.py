# Implemented by Dmitry Zinoviev <dzinoviev@suffolk.edu>

import numpy as np
import networkx as nx

def generalized_similarity(graph, min_eps=0.01, max_iter=50, weight=None):
    """
    Calculate generalized similarities between nodes in a BIPARTITE 
    graph, as described in [1]

    Parameters
    ----------
    G : graph
       A NetworkX graph

    min_eps : float, optional (default=0.01)
       Minimum attained precision.

    max_iter : int, optional (default=50)
       Maximal number of iterations.
    
    weight : string or None, optional (default='weight')
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    A tuple of two weighted graphs of similarities, actual attained 
    precision, and actual number of iterations.

    Notes
    -----
    Raises a ValueError exception if the graph is not bipartite.

    References
    ----------
    .. [1] Balázs Kovács, "A generalized model of relational similarity," 
           Social Networks, 32(3), July 2010, pp. 197–211.
    """

    if not nx.is_bipartite(graph):
        raise ValueError("Not a bipartite graph")

    s = nx.bipartite.sets(graph)
    
    arcs = nx.bipartite.biadjacency_matrix(graph, s[0], s[1],
                                           weight = weight).toarray()

    arcs0 = arcs - arcs.mean(axis=1)[:, np.newaxis]
    arcs1 = arcs.T - arcs.mean(axis=0)[:, np.newaxis]

    eps = min_eps + 1
    N = np.eye(arcs.shape[1])

    iters = 0
    
    while eps > min_eps and iters < max_iter:
        M = arcs0.dot(N).dot(arcs0.T)
        m = np.sqrt(M.diagonal())
        M = ((M / m).T / m).T
        
        Np = arcs1.dot(M).dot(arcs1.T)
        n = np.sqrt(Np.diagonal())
        Np = ((Np / n).T / n).T
        eps = np.abs(Np - N).max()
        N = Np

        iters += 1

    f = nx.relabel_nodes(nx.Graph(M), dict(enumerate(s[0])))
    g = nx.relabel_nodes(nx.Graph(Np), dict(enumerate(s[1])))

    return (f, g, eps, iters)
