
from rdflib import Graph
import networkx as nx

def load_rdf(ttl_paths):
    """
    Load one or more Turtle files into an rdflib Graph.
    :param ttl_paths: list of .ttl file paths
    :return: rdflib.Graph
    """
    g = Graph()
    for path in ttl_paths:
        g.parse(path, format="turtle")
    return g

def build_nx_graph(rdf_graph):
    """
    Convert an rdflib Graph into a NetworkX DiGraph.
    :param rdf_graph: rdflib.Graph
    :return: networkx.DiGraph
    """
    G = nx.DiGraph()
    for s, p, o in rdf_graph:
        G.add_edge(str(s), str(o), label=str(p))
    return G
