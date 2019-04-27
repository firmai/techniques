"""
Download data about food and nutrients from the USDA web site
and build a network of co-occurrences.
"""
import re
from urllib.request import urlopen
import networkx as nx
from bs4 import BeautifulSoup

BASE = 'https://ndb.nal.usda.gov/ndb/foods/show/{}'

G = nx.Graph()
for index in range(40001, 45001):
    print(index)
    try:
        data = urlopen(BASE.format(index)).read()
    except:
        print("Failed")
        continue
    soup = BeautifulSoup(data)
    try:
        ingredient_list = [x.strong.next_sibling for x
                           in soup.findAll('div', class_="col-md-12")
                           if x.strong
                           and x.strong.text.startswith('Ingredients')][0]
    except IndexError:
        print("Missing")
        continue
    ingredients = set(x.strip() for x
                      in re.split(r'[,()\[\]]', ingredient_list) if x)
    ingredients = set(x.replace('CONTAINS 2% OR LESS OF:', '') 
                      for x in ingredients)
    ingredients = set(x.replace('CONTAINS 2% OR LESS:', '') 
                      for x in ingredients)
    ingredients = set(x.strip('.*') for x in ingredients)
    ingredients = set(x.strip() for x in ingredients)
    G.add_edges_from((index, ing) for ing in ingredients)

G.remove_node('')
nx.write_graphml(G, open('usda-all.graphml', 'wb'))

# Create a bipartite projection
nodes = [n for n in G.nodes() if isinstance(n, str)]
F = nx.bipartite.weighted_projected_graph(G, nodes)
F.remove_edges_from(F.selfloop_edges())
F.remove_edges_from((e[0], e[1])
                    for e in F.edges(data=True)
                    if e[2]['weight'] < 5)
nx.write_graphml(F, open('usda.graphml', 'wb'))
