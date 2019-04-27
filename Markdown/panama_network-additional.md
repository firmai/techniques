
Update: I presented the content of this blog post at a Pydata meetup in Amsterdam. Other then adding a section on community detection, the presentation more or less follows this post. The slides can be found [here](http://www.degeneratestate.org/static/presentations/pppd2016.html).


Recently, the The International Consortium of Investigative Journalists (ICIJ) [released a dump](https://offshoreleaks.icij.org/pages/database) of some of the information they received as part of the [panama papers](https://panamapapers.icij.org/) leak.

The data released is in the form of a network: a collection of nodes which relate to entities, addresses, officers and intermediaries and a collection of edges which give information about the relationships between these nodes. For a full description of where the data comes from and what the fields mean see data/codebook.pdf in the [repository for this notebook](https://github.com/ijmbarr/panama-paper-network). 

A lot has been said about what is in the Panama Papers. Most of this has been focused around individuals who choose to use the business structures detailed in the leaks. In this post, I take a different look at the data, focusing on the structures that are implied by the Panama Papers, and on how we might be able to use ideas and tools from graph theory to explore these structures. 

My reason for this approach is that the current leak contains over 800,000 nodes and over 1.1 million relationships. Spending a minute looking at each relationship would take over two years, so  automation is the only way to begin to explore a dataset of this size. Automation however does have it's limitations - I am not an accountant or business lawyer, and I can't begin to speculate on the usefulness or even the interestingness of these results. My guess would be that this approach would need to be combined with both domain specific knowledge and local expertise on the people involved to get the most out of it.

This post is written as a jupyter notebook. This should allow anyone to reproduce my results. You can find the [repository for this notebook here](https://github.com/ijmbarr/panama-paper-network). Along with the analysis carried out in this notebook, I use a number of short, home build functions. These are also included in the repository.

Disclaimer: While I discuss several of the entities I find in the data, I am not accusing anyone of breaking the law.

## Creating a Graph

To begin with, I am going to load the nodes and edges into memory using pandas, normalising the names as I go:


```python
# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import random

%matplotlib inline
import matplotlib as mpl
mpl.style.use("ggplot")

%load_ext autoreload
%autoreload 2

from pputils import *
```


```python
"""
Build a "Panama" network using Pandas.
"""
import networkx as nx
import pandas as pd
import numpy as np

# Read the edge list and convert it to a network
edges = pd.read_csv("data/all_edges.csv")
edges = edges[edges["TYPE"] != "data/registered address"]
F = nx.from_pandas_dataframe(edges, "START_ID", "END_ID")
```

    /Users/dereksnow/anaconda/envs/py36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (4,5) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
# Read node lists
officers = pd.read_csv("data/Officers.csv", index_col="node_id")
intermediaries = pd.read_csv("data/Intermediaries.csv", index_col="node_id")
entities = pd.read_csv("data/Entities.csv", index_col="node_id")

# Combine the node lists into one dataframe
officers["type"] = "officer"
intermediaries["type"] = "intermediary"
entities["type"] = "entity"

all_nodes = pd.concat([officers, intermediaries, entities])

# Do some cleanup of names
all_nodes["name"] = all_nodes["name"].str.upper().str.strip()

# Ensure that all "Bearers" do not become a single node
all_nodes["name"].replace(
    to_replace=[r"MRS?\.\s+", r"\.", r"\s+", "LIMITED", "THE BEARER",
                 "BEARER", "BEARER 1", "EL PORTADOR", "AL PORTADOR"],
    value=["", "", " ", "LTD", np.nan, np.nan, np.nan, np.nan, np.nan],
    inplace=True, regex=True)

# The network is ready to use!
# As an exercise, let's have a look at some assets
CCODES = "UZB", "TKM", "KAZ", "KGZ", "TJK"
seeds = all_nodes[all_nodes["country_codes"].isin(CCODES)].index
nodes_of_interest = set.union(*[\
        set(nx.single_source_shortest_path_length(F, seed, cutoff=2).keys())
        for seed in seeds])

# Extract the subgraph and relabel it
ego = nx.subgraph(F, nodes_of_interest)

nodes = all_nodes.ix[ego]
nodes = nodes[~nodes.index.duplicated()]
nx.set_node_attributes(ego, "cc", nodes["country_codes"])
valid_names = nodes[nodes["name"].notnull()]["name"].to_dict()
nx.relabel_nodes(ego, valid_names, copy=False)

# Save and proceed to Gephi
with open("panama-ca.graphml", "wb") as ofile:
    nx.write_graphml(ego, ofile)
```

    /Users/dereksnow/anaconda/envs/py36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (16) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    /Users/dereksnow/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel/__main__.py:33: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated



```python
"""
Analyze "Panama Papers"
"""
import csv
import pickle
import itertools
from collections import Counter
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from pputils import dzcnapy_plotlib as dzcnapy

EDGES = "beneficiary"
NODES = (("data/Entities.csv", "jurisdiction", "name"),
         ("data/Officers.csv", "country_codes", "name"),
         ("data/Intermediaries.csv", "country_codes", "name"))

panama = nx.Graph()

with open("data/all_edges.csv") as infile:
    data = csv.DictReader(infile)
    panama.add_edges_from((link["START_ID"], link["END_ID"])
                          for link in data
                          if link["TYPE"].lower().startswith(EDGES))


nodes = set(panama.nodes())
relabel = {}

for f, cc, name in NODES:
    with open(f) as infile:
        kind = f.split(".")[0]
        data = csv.DictReader(infile)
        names_countries = {node["node_id"] :
                           (node[name].strip().upper(), node[cc])
                           for node in data
                           if node["node_id"] in nodes}
    names =     {nid: values[0] for nid, values in names_countries.items()}
    countries = {nid: values[1] for nid, values in names_countries.items()}
    kinds =     {nid: kind      for nid, _      in names_countries.items()}
    nx.set_node_attributes(panama, "country", countries)
    nx.set_node_attributes(panama, "kind", kinds)
    relabel.update(names)

nx.relabel_nodes(panama, relabel, copy=False)

if "ISSUES OF:" in panama:
    panama.remove_node("ISSUES OF:")

if "" in panama:
    panama.remove_node("")

print(nx.number_of_nodes(panama), nx.number_of_edges(panama))

components = [p.nodes() for p in nx.connected_component_subgraphs(panama)
              if nx.number_of_nodes(p) >= 20
              or nx.number_of_edges(p) >= 20]
panama0 = panama.subgraph(itertools.chain.from_iterable(components))

print(nx.number_of_nodes(panama0), nx.number_of_edges(panama0))

with open("panama-beneficiary.pickle", "wb") as outfile:
    pickle.dump(panama0, outfile)

cdict = {"Entities": "pink", "Officers": "blue", 
         "Intermediaries" : "green"}
c = [cdict[panama0.node[n]["kind"]] for n in panama0]
dzcnapy.small_attrs["node_color"] = c
pos = graphviz_layout(panama0)
nx.draw_networkx(panama0, pos=pos, with_labels=False, **dzcnapy.small_attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("panama0")

nx.attribute_assortativity_coefficient(panama0, "kind")
nx.attribute_mixing_matrix(panama0, "kind",
                           mapping={"Entities": 0, "Officers": 1,
                                    "Intermediaries" : 2})
nx.attribute_assortativity_coefficient(panama0, "country")
nx.degree_assortativity_coefficient(panama0)

deg = nx.degree(panama0)
x, y = zip(*Counter(deg.values()).items())

plt.scatter(x, y, s=100, c="pink")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlim(0.9, max(x))
plt.ylim(0.9, max(y))
plt.xlabel("Degree")
plt.ylabel("Frequency")
dzcnapy.plot("panama-beneficiaries")

top10 = sorted([(n, panama0.node[n]["kind"], v) for n, v in deg.items()],
               key=lambda x: x[2], reverse=True)[:10]
print("\n".join(["{} ({}): {}".format(*t) for t in top10]))

```

    0 0
    0 0



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-4-4159c0f5f220> in <module>()
         69 pos = graphviz_layout(panama0)
         70 nx.draw_networkx(panama0, pos=pos, with_labels=False, **dzcnapy.small_attrs)
    ---> 71 dzcnapy.set_extent(pos, plt)
         72 dzcnapy.plot("panama0")
         73 


    /Volumes/extra/FirmAI/Networks/panama-paper-network-master/pputils/dzcnapy_plotlib.py in set_extent(positions, axes, title)
         37         axes.set_title(title)
         38 
    ---> 39     x_values, y_values = zip(*positions.values())
         40     x_max = max(x_values)
         41     y_max = max(y_values)


    ValueError: not enough values to unpack (expected 2, got 0)



![png](panama_network-additional_files/panama_network-additional_4_2.png)



```python
# load the raw data into dataframes and cleans up some of the strings
adds = pd.read_csv("data/Addresses.csv", low_memory=False)

ents = pd.read_csv("data/Entities.csv", low_memory=False)
ents["name"] = ents.name.apply(normalise)

inter = pd.read_csv("data/Intermediaries.csv", low_memory=False)
inter["name"] = inter.name.apply(normalise)

offi = pd.read_csv("data/Officers.csv", low_memory=False)
offi["name"] = offi.name.apply(normalise)

edges = pd.read_csv("data/all_edges.csv", low_memory=False)
```

We can now build the graph. I am using the [networkx](https://networkx.github.io/) library to represent the network. I use the node_id property to represent the node, all other information provided by the files is stored in the nodes details.

I am treating the graph as directed, as the relationships implied by the edges are directional (e.g. "shareholder of" or "director of"), however for part of the analysis we will switch to an undirected form.


```python
ents.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>name</th>
      <th>jurisdiction</th>
      <th>jurisdiction_description</th>
      <th>country_codes</th>
      <th>countries</th>
      <th>incorporation_date</th>
      <th>inactivation_date</th>
      <th>struck_off_date</th>
      <th>closed_date</th>
      <th>ibcRUC</th>
      <th>status</th>
      <th>company_type</th>
      <th>service_provider</th>
      <th>sourceID</th>
      <th>valid_until</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000001</td>
      <td>tiansheng industry and trading co., ltd.</td>
      <td>SAM</td>
      <td>Samoa</td>
      <td>HKG</td>
      <td>Hong Kong</td>
      <td>23-MAR-2006</td>
      <td>18-FEB-2013</td>
      <td>15-FEB-2013</td>
      <td>NaN</td>
      <td>25221</td>
      <td>Defaulted</td>
      <td>NaN</td>
      <td>Mossack Fonseca</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000002</td>
      <td>ningbo sunrise enterprises united co., ltd.</td>
      <td>SAM</td>
      <td>Samoa</td>
      <td>HKG</td>
      <td>Hong Kong</td>
      <td>27-MAR-2006</td>
      <td>27-FEB-2014</td>
      <td>15-FEB-2014</td>
      <td>NaN</td>
      <td>25249</td>
      <td>Defaulted</td>
      <td>NaN</td>
      <td>Mossack Fonseca</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000003</td>
      <td>hotfocus co., ltd.</td>
      <td>SAM</td>
      <td>Samoa</td>
      <td>HKG</td>
      <td>Hong Kong</td>
      <td>10-JAN-2006</td>
      <td>15-FEB-2012</td>
      <td>15-FEB-2012</td>
      <td>NaN</td>
      <td>24138</td>
      <td>Defaulted</td>
      <td>NaN</td>
      <td>Mossack Fonseca</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000004</td>
      <td>sky-blue gifts &amp; toys co., ltd.</td>
      <td>SAM</td>
      <td>Samoa</td>
      <td>HKG</td>
      <td>Hong Kong</td>
      <td>06-JAN-2006</td>
      <td>16-FEB-2009</td>
      <td>15-FEB-2009</td>
      <td>NaN</td>
      <td>24012</td>
      <td>Defaulted</td>
      <td>NaN</td>
      <td>Mossack Fonseca</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000005</td>
      <td>fortunemaker investments corporation</td>
      <td>SAM</td>
      <td>Samoa</td>
      <td>HKG</td>
      <td>Hong Kong</td>
      <td>19-APR-2006</td>
      <td>15-MAY-2009</td>
      <td>15-FEB-2008</td>
      <td>NaN</td>
      <td>R25638</td>
      <td>Changed agent</td>
      <td>NaN</td>
      <td>Mossack Fonseca</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
adds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>name</th>
      <th>address</th>
      <th>country_codes</th>
      <th>countries</th>
      <th>sourceID</th>
      <th>valid_until</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14000001</td>
      <td>NaN</td>
      <td>-\t27 ROSEWOOD DRIVE #16-19 SINGAPORE 737920</td>
      <td>SGP</td>
      <td>Singapore</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14000002</td>
      <td>NaN</td>
      <td>"Almaly Village" v.5, Almaty Kazakhstan</td>
      <td>KAZ</td>
      <td>Kazakhstan</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14000003</td>
      <td>NaN</td>
      <td>"Cantonia" South Road St Georges Hill Weybridg...</td>
      <td>GBR</td>
      <td>United Kingdom</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14000004</td>
      <td>NaN</td>
      <td>"CAY-OS" NEW ROAD; ST.SAMPSON; GUERNSEY; CHANN...</td>
      <td>GGY</td>
      <td>Guernsey</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14000005</td>
      <td>NaN</td>
      <td>"Chirag" Plot No 652; Mwamba Road; Kizingo; Mo...</td>
      <td>KEN</td>
      <td>Kenya</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
inter.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>name</th>
      <th>country_codes</th>
      <th>countries</th>
      <th>status</th>
      <th>sourceID</th>
      <th>valid_until</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11000001</td>
      <td>michael papageorge, mr.</td>
      <td>ZAF</td>
      <td>South Africa</td>
      <td>ACTIVE</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11000002</td>
      <td>corfiducia anstalt</td>
      <td>LIE</td>
      <td>Liechtenstein</td>
      <td>ACTIVE</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11000003</td>
      <td>david, ronald</td>
      <td>MCO</td>
      <td>Monaco</td>
      <td>SUSPENDED</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11000004</td>
      <td>de  boutselis, jean-pierre</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>SUSPENDED</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11000005</td>
      <td>the levant lawyers (tll)</td>
      <td>LBN</td>
      <td>Lebanon</td>
      <td>ACTIVE</td>
      <td>Panama Papers</td>
      <td>The Panama Papers  data is current through 2015</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
edges.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>START_ID</th>
      <th>TYPE</th>
      <th>END_ID</th>
      <th>link</th>
      <th>start_date</th>
      <th>end_date</th>
      <th>sourceID</th>
      <th>valid_until</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000035</td>
      <td>registered_address</td>
      <td>14095990</td>
      <td>registered address</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000044</td>
      <td>registered_address</td>
      <td>14091035</td>
      <td>registered address</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000055</td>
      <td>registered_address</td>
      <td>14095990</td>
      <td>registered address</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000064</td>
      <td>registered_address</td>
      <td>14091429</td>
      <td>registered address</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000089</td>
      <td>registered_address</td>
      <td>14098253</td>
      <td>registered address</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create graph

G = nx.DiGraph()

for n,row in adds.iterrows():
    G.add_node(row.node_id, node_type="address", details=row.to_dict())
    
for n,row in ents.iterrows():
    G.add_node(row.node_id, node_type="entities", details=row.to_dict())
    
for n,row in inter.iterrows():
    G.add_node(row.node_id, node_type="intermediates", details=row.to_dict())
    
for n,row in offi.iterrows():
    G.add_node(row.node_id, node_type="officers", details=row.to_dict())
    
for n,row in edges.iterrows():
    G.add_edge(row.START_ID, row.END_ID, rel_type=row.TYPE, details={})
```


```python
# store locally to allow faster loading
nx.write_adjlist(G,"pp_graph.adjlist")

# G = nx.read_adjlist("pp_graph.adjlist")
```

The first thing we are going to want to do is merge similar names into the same node:

## I was not able to merge - advice elswehere


```python
from pputils import *

print(G.number_of_nodes())
print(G.number_of_edges())

merge_similar_names(G)

print(G.number_of_nodes())
print(G.number_of_edges())
```

    559600
    657488
    559600
    657488


## Subgraphs

One of the first questions we can ask about the network is whether it is connected. Two nodes are considered connected if there is a path between the nodes. Networkx allows us to do this directly by splitting the graph into connected sub-graphs:


```python
subgraphs = [g for g in nx.connected_component_subgraphs(G.to_undirected())]
```


```python
subgraphs = sorted(subgraphs, key=lambda x: x.number_of_nodes(), reverse=True)
print([s.number_of_nodes() for s in subgraphs[:10]])
```

    [455479, 2995, 730, 644, 597, 536, 409, 406, 380, 378]


It looks like the majority of nodes are all connected into one large connected graph, which contains nearly 90% of all the nodes. We will look at this graph soon, but to get a feeling for what information is contained within these graphs, let's plot a few of the smaller ones:


```python
## interesting, this function just exists
plot_graph(subgraphs[134])
```


![png](panama_network-additional_files/panama_network-additional_20_0.png)


In this graph we are seeing one intermediate "studia notanstefando", acting as the intermediate for a number of entities, in this case what look like companies. You can also tell how crowded the graph is becoming. We are going to see this problem just gets worse as graph sizes grow and at some point the data becomes impossible to visualise in a concise manner.

Let us take a look at a more complex example: 


```python
plot_graph(subgraphs[210], figsize=(8,8))
```


![png](panama_network-additional_files/panama_network-additional_22_0.png)




## The Main Network
Turning our attention to that largest connected sub-graph, we run into problems. The graph is far too big to consider plotting it and analysing it meaningfully by eye. Instead we need to try and phase our questions in such a way that the computer does the work for us.

From the graphs we saw above, it looks like the intermediaries tend to sit at the centre of things. Does this hold true in the large graph? To test this, we can find the average degree of each node type, where "degree" is the number of edges connected to a node.


```python
# grab the largest subgraph
g = subgraphs[0]
```


```python
# look at node degree
nodes = g.nodes()
g_degree = g.degree()
types = [g.node[n]["node_type"] for n in nodes]
degrees = [g_degree[n] for n in nodes]
names = [get_node_label(g.node[n]) for n in nodes]
node_degree = pd.DataFrame(data={"node_type":types, "degree":degrees, "name": names}, index=nodes)
```


```python
# how many by node_type - Degree is how many nodes
node_degree.groupby("node_type").agg(["count", "mean", "median"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">degree</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>node_type</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>address</th>
      <td>81937</td>
      <td>1.638222</td>
      <td>1</td>
    </tr>
    <tr>
      <th>entities</th>
      <td>169676</td>
      <td>2.509129</td>
      <td>2</td>
    </tr>
    <tr>
      <th>intermediates</th>
      <td>3096</td>
      <td>54.804910</td>
      <td>7</td>
    </tr>
    <tr>
      <th>officers</th>
      <td>200770</td>
      <td>1.924262</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the _median_ values of each group aren't that different - 50% of most nodes have only a few edges connected to them. However the large _mean_ of the degree of intermediates suggests that the distribution is highly uneven and long tailed where there are a small number intermediaries who have a large number of the edges.

We can check this by looking at the nodes ten with the largest degree


```python
node_degree.sort_values("degree", ascending=False)[0:15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree</th>
      <th>name</th>
      <th>node_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11001746</th>
      <td>7016</td>
      <td>orion house services (hk) limited</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11011863</th>
      <td>4364</td>
      <td>mossack fonseca &amp; co.</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11012037</th>
      <td>4117</td>
      <td>prime corporate solutions sarl</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11001708</th>
      <td>4094</td>
      <td>offshore business consultant (int'l) limited</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11008027</th>
      <td>3888</td>
      <td>mossack fonseca &amp; co. (singapore) pte ltd.</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>12160432</th>
      <td>3883</td>
      <td>mossfon subscribers ltd.</td>
      <td>officers</td>
    </tr>
    <tr>
      <th>11009351</th>
      <td>3168</td>
      <td>consulco international limited</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11011539</th>
      <td>2538</td>
      <td>mossack fonseca &amp; co. (u.k.) limited</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11009139</th>
      <td>2055</td>
      <td>mossack fonseca &amp; co. (peru) corp.</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11009218</th>
      <td>2045</td>
      <td>power point int'l co., ltd.</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11010643</th>
      <td>2014</td>
      <td>legal consulting services limited</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11012290</th>
      <td>1871</td>
      <td>mossfon managers ltd.</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11006103</th>
      <td>1659</td>
      <td>experta corporate &amp; trust services</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11012118</th>
      <td>1550</td>
      <td>mossack fonseca &amp; co. cz, s.r.o</td>
      <td>intermediates</td>
    </tr>
    <tr>
      <th>11010502</th>
      <td>1479</td>
      <td>rawi &amp; co.</td>
      <td>intermediates</td>
    </tr>
  </tbody>
</table>
</div>



The next few intermediates that appear are "mossack fonseca & co", "prime corporate solutions sarl", "offshore business consultant (int'l) limited" and "sealight incorporations limited". 

Given that the Intermediary appears to be a middleman that helps create the entities, it is easy to consider that each one could be linked to many entities. What isn't immediately clear is how they might be linked together. Let's take a look at the shortest path between "mossack fonseca & co" and "prime corporate solutions sarl":


```python
def plot_path(g, path):
    plot_graph(g.subgraph(path), label_edges=True)

path = nx.shortest_path(g, source=11011863, target=11012037)
plot_path(G, path)
```


![png](panama_network-additional_files/panama_network-additional_30_0.png)


It seems that the two intermediaries are linked together through companies who share a common director, ["mossfon subscribers ltd."]. As it’s name suggests, it also acts as director for a number of other companies:


```python
offi[offi["name"]=="mossfon subscribers ltd."]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>name</th>
      <th>country_codes</th>
      <th>countries</th>
      <th>sourceID</th>
      <th>valid_until</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10293</th>
      <td>12012808</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10832</th>
      <td>12013347</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22346</th>
      <td>12024861</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27579</th>
      <td>12030094</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44340</th>
      <td>12046855</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44524</th>
      <td>12047039</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44928</th>
      <td>12047443</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45457</th>
      <td>12047972</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45532</th>
      <td>12048047</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81325</th>
      <td>12083840</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81394</th>
      <td>12083909</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81500</th>
      <td>12084015</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>81696</th>
      <td>12084211</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>129619</th>
      <td>12161143</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>144555</th>
      <td>12145825</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>152334</th>
      <td>12153441</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>152480</th>
      <td>12153442</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>154656</th>
      <td>12155567</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>159521</th>
      <td>12160373</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>159573</th>
      <td>12160432</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>160257</th>
      <td>12161142</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>160310</th>
      <td>12161185</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>168935</th>
      <td>12170434</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>169453</th>
      <td>12171085</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>170366</th>
      <td>12171919</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>171534</th>
      <td>12173175</td>
      <td>mossfon subscribers ltd.</td>
      <td>WSM</td>
      <td>Samoa</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>174080</th>
      <td>12175800</td>
      <td>mossfon subscribers ltd.</td>
      <td>PAN</td>
      <td>Panama</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>223110</th>
      <td>12000555</td>
      <td>mossfon subscribers ltd.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Panama Papers</td>
      <td>The Panama Papers data is current through 2015</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_graph(G.subgraph(nx.ego_graph(g, 12046855, radius=1).nodes()), label_edges=True)
```


![png](panama_network-additional_files/panama_network-additional_33_0.png)


We can do the same for, say, "mossack fonseca & co." and "sealight incorporations limited":


```python
path = nx.shortest_path(g,11011863, 298293)
plot_path(G, path)
```


![png](panama_network-additional_files/panama_network-additional_35_0.png)


This chain is more convoluted, but it looks like a series of companies tied together by common shareholders or directors.

## Degree Distribution

We can also ask how the degree of the graph is distributed.


```python
max_bin = max(degrees)
n_bins = 20
log_bins = [10 ** ((i/n_bins) * np.log10(max_bin)) for i in range(0,n_bins)]
fig, ax = plt.subplots()
node_degree.degree.value_counts().hist(bins=log_bins,log=True)
ax.set_xscale('log')

plt.xlabel("Number of Nodes")
plt.ylabel("Number of Degrees")
plt.title("Distribution of Degree");
```


![png](panama_network-additional_files/panama_network-additional_37_0.png)


If we squint, it might look like a power law distribution, giving a [scale free graph](https://en.wikipedia.org/wiki/Scale-free_network). But we'd have to be squinting.

The main result is that the distribution is long tailed - a small number of nodes are involved in most of the links.

## Node Importance

We are starting to explore how entities are connected together. Intuitively, you might expect nodes with a high degree to be the most "important" - that they sit at the centre of the graph and are closely linked to every other node. However, other measures exist.

A common measure for importance of a node is its [page rank](https://en.wikipedia.org/wiki/PageRank). Page rank is one of the measures used by google to determine the importance of a webpage, and is named after Larry Page. Essentially, if we were to perform a random walk through a graph, jumping to a random page every now and then, the time spent on each node is proportional to its page-rank.

We can calculate the page rank for each node below, and look at the top ranked nodes:


```python
%time pr = nx.pagerank_scipy(g)
```

    CPU times: user 4.06 s, sys: 0 ns, total: 4.06 s
    Wall time: 4.06 s



```python
node_degree["page_rank"] = node_degree.index.map(lambda x: pr[x])
```


```python
node_degree.sort_values("page_rank", ascending=False)[0:15]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree</th>
      <th>name</th>
      <th>node_type</th>
      <th>page_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>236724</th>
      <td>37329</td>
      <td>Portcullis TrustNet Chambers P.O. Box 3444 Roa...</td>
      <td>address</td>
      <td>0.007766</td>
    </tr>
    <tr>
      <th>54662</th>
      <td>36115</td>
      <td>portcullis trustnet (bvi) limited</td>
      <td>officers</td>
      <td>0.007553</td>
    </tr>
    <tr>
      <th>11001746</th>
      <td>7014</td>
      <td>orion house services (hk) limited</td>
      <td>intermediates</td>
      <td>0.002151</td>
    </tr>
    <tr>
      <th>11001708</th>
      <td>4094</td>
      <td>offshore business consultant (int'l) limited</td>
      <td>intermediates</td>
      <td>0.001420</td>
    </tr>
    <tr>
      <th>11012037</th>
      <td>4112</td>
      <td>prime corporate solutions sarl</td>
      <td>intermediates</td>
      <td>0.001271</td>
    </tr>
    <tr>
      <th>11008027</th>
      <td>3887</td>
      <td>mossack fonseca &amp; co. (singapore) pte ltd.</td>
      <td>intermediates</td>
      <td>0.001180</td>
    </tr>
    <tr>
      <th>96909</th>
      <td>4253</td>
      <td>portcullis trustnet (samoa) limited</td>
      <td>officers</td>
      <td>0.001013</td>
    </tr>
    <tr>
      <th>12174256</th>
      <td>3885</td>
      <td>mossfon suscribers ltd.</td>
      <td>officers</td>
      <td>0.000963</td>
    </tr>
    <tr>
      <th>11009139</th>
      <td>2036</td>
      <td>mossack fonseca &amp; co. (peru) corp.</td>
      <td>intermediates</td>
      <td>0.000908</td>
    </tr>
    <tr>
      <th>11011863</th>
      <td>4356</td>
      <td>mossack fonseca &amp; co.</td>
      <td>intermediates</td>
      <td>0.000759</td>
    </tr>
    <tr>
      <th>264051</th>
      <td>2671</td>
      <td>Company Kit Limited Unit A, 6/F Shun On Comm B...</td>
      <td>address</td>
      <td>0.000749</td>
    </tr>
    <tr>
      <th>297687</th>
      <td>2671</td>
      <td>company kit limited</td>
      <td>intermediates</td>
      <td>0.000749</td>
    </tr>
    <tr>
      <th>288469</th>
      <td>5697</td>
      <td>Unitrust Corporate Services Ltd. John Humphrie...</td>
      <td>address</td>
      <td>0.000741</td>
    </tr>
    <tr>
      <th>298333</th>
      <td>5695</td>
      <td>unitrust corporate services ltd.</td>
      <td>intermediates</td>
      <td>0.000740</td>
    </tr>
    <tr>
      <th>294268</th>
      <td>3329</td>
      <td>offshore business consultant (hk) ltd.</td>
      <td>intermediates</td>
      <td>0.000666</td>
    </tr>
  </tbody>
</table>
</div>



As it turns out, page rank picks out similar nodes to looking at degree. 

If I were interested in identifying the main players in setting up offshore companies, these are the intermediates that I would start looking at first.

So what happens if we look at the page rank, but just for entities?


```python
node_degree[node_degree.node_type == "entities"].sort_values("page_rank", ascending=False)[0:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree</th>
      <th>name</th>
      <th>node_type</th>
      <th>page_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10200346</th>
      <td>998</td>
      <td>accelonic ltd.</td>
      <td>entities</td>
      <td>0.000568</td>
    </tr>
    <tr>
      <th>137067</th>
      <td>440</td>
      <td>hannspree inc.</td>
      <td>entities</td>
      <td>0.000249</td>
    </tr>
    <tr>
      <th>153845</th>
      <td>322</td>
      <td>m.j. health management international holding inc.</td>
      <td>entities</td>
      <td>0.000178</td>
    </tr>
    <tr>
      <th>10133161</th>
      <td>432</td>
      <td>dale capital group limited</td>
      <td>entities</td>
      <td>0.000160</td>
    </tr>
    <tr>
      <th>10154669</th>
      <td>242</td>
      <td>magn development limited</td>
      <td>entities</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>10126705</th>
      <td>203</td>
      <td>digiwin systems group holding limited</td>
      <td>entities</td>
      <td>0.000114</td>
    </tr>
    <tr>
      <th>10136878</th>
      <td>147</td>
      <td>mulberry holdings asset limited</td>
      <td>entities</td>
      <td>0.000076</td>
    </tr>
    <tr>
      <th>10204952</th>
      <td>158</td>
      <td>rockover resources limited</td>
      <td>entities</td>
      <td>0.000074</td>
    </tr>
    <tr>
      <th>10103570</th>
      <td>493</td>
      <td>vela gas investments ltd.</td>
      <td>entities</td>
      <td>0.000074</td>
    </tr>
    <tr>
      <th>176625</th>
      <td>449</td>
      <td>wan chi investments limited</td>
      <td>entities</td>
      <td>0.000071</td>
    </tr>
  </tbody>
</table>
</div>




```python
t = nx.ego_graph(g, 10165699, radius=1)
plot_graph(t, label_edges=True)
```


![png](panama_network-additional_files/panama_network-additional_44_0.png)



It looks like we just end up with the entities that have lots of shareholders and who use one of the high ranking intermediates.


# Clustering

Another measurement we can make of the "shape" of a graph is its [clustering coefficient](https://en.wikipedia.org/wiki/Clustering_coefficient). For each node, this measures how connected its neighbours are with each other. You can think of it as a measure of the local structure of the graph: what fraction of a nodes neighbours are also neighbours of each other.


```python
%time cl = nx.clustering(g)
```

    CPU times: user 4min 13s, sys: 52 ms, total: 4min 13s
    Wall time: 4min 13s



```python
node_degree["clustering_coefficient"] = node_degree.index.map(lambda x: cl[x])
```


```python
node_degree.clustering_coefficient.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd6242f8d30>




![png](panama_network-additional_files/panama_network-additional_48_1.png)



```python
node_degree.sort_values(["clustering_coefficient", "degree"], ascending=False)[0:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree</th>
      <th>name</th>
      <th>node_type</th>
      <th>page_rank</th>
      <th>clustering_coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>122671</th>
      <td>3</td>
      <td></td>
      <td>officers</td>
      <td>9.695260e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>122762</th>
      <td>3</td>
      <td>sharecorp limited</td>
      <td>officers</td>
      <td>7.886701e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26098</th>
      <td>2</td>
      <td>david john bird</td>
      <td>officers</td>
      <td>9.492465e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>39673</th>
      <td>2</td>
      <td>axisinvest corporation</td>
      <td>officers</td>
      <td>9.615074e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>41341</th>
      <td>2</td>
      <td>healthcare lifestyle holdings limited</td>
      <td>officers</td>
      <td>5.569585e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>41363</th>
      <td>2</td>
      <td>key enrichment limited</td>
      <td>officers</td>
      <td>5.543170e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>41378</th>
      <td>2</td>
      <td>chuen tat overseas limited</td>
      <td>officers</td>
      <td>7.100476e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>41386</th>
      <td>2</td>
      <td>woodwind development limited</td>
      <td>officers</td>
      <td>6.078500e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>41437</th>
      <td>2</td>
      <td>tonga group services limited</td>
      <td>officers</td>
      <td>7.430761e-07</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>41438</th>
      <td>2</td>
      <td>millennium media group limited</td>
      <td>officers</td>
      <td>8.066175e-07</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



As it turns out, there isn't much structure. Most nodes have clustering coefficients of zero. The few that have non-zero values tend to have low degrees. This means that the panama paper network isn't an example of a small world network. To see what's happening in the few non-zero cases, we can look at an example sub-graph below:


```python
t = nx.ego_graph(g, 122762, radius=1)
plot_graph(G.subgraph(t), label_edges=True)
```


![png](panama_network-additional_files/panama_network-additional_51_0.png)


In this case, it looks like it is just due to a shared address between "sharecorp limited" and "bizlink network incorporated", and some confusion over the multiple occurrences of "sharecorp".

As a side note, I'm curious how these businesses come up with their names - I don't know anything about sharecorp limited, but it sounds like a name used as an example in economics textbooks.

## Ownership

So far we have looked at the fully connected graph, even with connections like "address of" and "intermediary of". While this does tell us that there has been nearly 40,000 businesses registered to a single address, we might want to confine ourselves to just looking at the network formed where there is some form of ownership.

Unlike our previous graph, we are going to make this one directed - this mean that each edge has a direction associated with it. For example the relationship "shareholder of" acts in one direction.

I've collected together all the relationships I think involve some kind of ownership, but I am not a lawyer or accountant, so these may be wrong.


```python
owner_rels = set({
    'shareholder of',
    'Shareholder of',
    'Director / Shareholder of',
    'Director of',
    'Director (Rami Makhlouf) of',
    'Power of Attorney of',
    'Director / Shareholder / Beneficial Owner of',
    'Member / Shareholder of',
    'Owner of',
    'Beneficial Owner of',
    'Power of attorney of',
    'Owner, director and shareholder of',
    'President - Director of',
    'Sole shareholder of',
    'President and director of',
    'Director / Beneficial Owner of',
    'Power of Attorney / Shareholder of',
    'Director and shareholder of',
    'beneficiary of',
    'President of',
    'Member of Foundation Council of',
    'Beneficial owner of',
    'Sole signatory of',
    'Sole signatory / Beneficial owner of',
    'Principal beneficiary of',
    'Protector of',
    'Beneficiary, shareholder and director of',
    'Beneficiary of',
    'Shareholder (through Julex Foundation) of',
    'First beneficiary of',
    'Authorised Person / Signatory of',
    'Successor Protector of',
    'Register of Shareholder of',
    'Reserve Director of',
    'Resident Director of',
    'Alternate Director of',
    'Nominated Person of',
    'Register of Director of',
    'Trustee of Trust of',
    'Personal Directorship of',
    'Unit Trust Register of',
    'Chairman of',
    'Board Representative of',
    'Custodian of',
    'Nominee Shareholder of',
    'Nominee Director of',
    'Nominee Protector of',
    'Nominee Investment Advisor of',
    'Nominee Trust Settlor of',
    'Nominee Beneficiary of',
    'Nominee Secretary of',
    'Nominee Beneficial Owner of'
});
```


```python
# copy main graph
g2 = G.copy()

# remove non-ownership edges
for e in g2.edges(data=True):
    if e[2]["rel_type"] not in owner_rels:
        g2.remove_edge(e[0], e[1])
        
# get all subgraphs
subgraphs = [sg for sg in nx.connected_component_subgraphs(g2.to_undirected())]
subgraphs = sorted(subgraphs, key=lambda x: x.number_of_nodes(), reverse=True)
len(subgraphs)
```




    401655




```python
g2.number_of_edges()
```




    465646



Removing two thirds of the nodes breaks this graph into lots of smaller sub-graphs. Most of these graphs are uninteresting and simply reflect that one company is owned by a large number of shareholders. Consider the graph below:

(Note: we are now looking at a "directed" graph. The edges are slightly wider at one end to represent their directionality)


```python
tt = subgraphs[1000].nodes()
plot_graph(g2.subgraph(tt), label_edges=True)
```


![png](panama_network-additional_files/panama_network-additional_57_0.png)


To identify more interesting structures, we can look at sub-graphs with the largest median node degree:


```python
avg_deg = pd.Series(data=[np.median(list(sg.degree().values())) for sg in subgraphs],
                    index=range(0,len(subgraphs)))
```


```python
avg_deg.sort_values(ascending=False)[0:10]
```




    790     6.0
    582     6.0
    268     5.0
    2643    5.0
    2040    5.0
    263     4.5
    1420    4.0
    1904    4.0
    745     4.0
    3271    4.0
    dtype: float64




```python
tt = subgraphs[582].nodes()
plot_graph(g2.subgraph(tt))
```


![png](panama_network-additional_files/panama_network-additional_61_0.png)


In these cases we are looking at a small group of companies that share the same owners.

## The Longest Line

We can also ask what the longest chain of ownership links is:


```python
lp = nx.dag_longest_path(g2)
print("The longest path is {} nodes long.".format(len(lp)))
plot_graph(g2.subgraph(lp), label_edges=True)
```

    The longest path is 4 nodes long.



![png](panama_network-additional_files/panama_network-additional_63_1.png)


It is surprisingly short.

And with that, I have finished my explorations.

I'd like to reiterate that without prior knowledge of this sort of network, it is hard to know what constitutes an "interesting" business/entities structure, or what might be the sign of potentially criminal/immoral behaviour. My guess is that no single measure captures it completely, but rather we could combine multiple measures, and some machine learning to automatically identify those areas for future inquiry. In fact, I'd be surprised if this wasn't already being done somewhere. 

That's about all I have to say for now on this network. I'm sure there is more that can be done, and interesting questions that can be asked and answered, but I lack the background required to know where to start. 

