import pandas as pd
import networkx as nx


PUNCTUATION = """.,"'()[]{}:;/!£$%^&*-="""

SAME_NAME_REL_TYPES = [
    'similar name and address as',
    'same name and registration date as',
    'same address as',
]


def normalise(s, strip_punctuation=False):
    """
    Normalises the format of a string.
    Parameters
    ----------
    s: str
    Returns:
    -------
    """
    if pd.isnull(s):
        return ""

    s = s.strip().lower()

    if strip_punctuation:
        for c in PUNCTUATION:
            s = s.replace(c, "")

    return s


def get_node_by_name(g, name):
    """
    Returns nodes whose name contains "name"
    :param g: Graph
    :param name: str
    :return: node
    """
    def contains_name(node):
        n_data = node[1]
        if n_data["node_type"] != "address":
            if name in n_data["details"]["name"]:
                return True
        return False

    return get_node_by_predicate(g, contains_name)


def get_node_by_predicate(g, pred):
    """
    Returns nodes whose name contains "name"
    :param g: Graph
    :param pred: Function node -> Bool
    :return: node
    """
    nodes_to_return = []
    for node in g.nodes(data=True):
        if pred(node):
            nodes_to_return.append(node)
    return nodes_to_return


def get_edges_for_node(g, n):
    """
    :param g: Graph
    :param n: node_id
    :return: list(edges)
    """
    return [e for e in g.edges(data=True)
            if e[0] == n
            or e[1] == n]


def get_node_label(n):
    if n["node_type"] == "address":
        if pd.isnull(n["details"]["address"]):
            return ""
        return n["details"]["address"].replace(";", "\n")
    return n["details"]["name"]


def merge_edge(g, target_edge):

    n_remove, n_replace = target_edge[0:2]

    edges_to_replace = g.edges(nbunch=n_remove, data=True)

    new_edges = []

    for e in edges_to_replace:
        if (e[0],e[1]) == (n_remove, n_replace):
            continue
        if e[0] == n_remove:
            new_edges.append( (n_replace, e[1], e[2]) )
        else:
            new_edges.append( (e[0], n_replace, e[2]) )

    g.remove_node(n_remove)

    for e in new_edges:
        g.add_edge(e[0], e[1], e[2])

    return new_edges


def merge_similar_names(g):

    edges = g.edges(data=True)
    removed = set()

    while edges:
        current_edge = edges.pop()

        if current_edge[2]["TYPE"] not in SAME_NAME_REL_TYPES:
            continue

        if current_edge[0] in removed or current_edge[1] in removed:
            continue

        new_edges = merge_edge(g, current_edge)

        edges += new_edges
        removed.add(current_edge[0])
    return g


def to_directed(g):
    dg = nx.DiGraph()

    for n in g.nodes(data=True):
        dg.add_node(n[0], attr_dict=n[1])

    for e in g.edges(data=True):
        dg.add_edge(e[0], e[1], attr_dict=e[2])

    return dg