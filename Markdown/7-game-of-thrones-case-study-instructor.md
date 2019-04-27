
##### Let's change gears and talk about Game of thrones or shall I say Network of Thrones.

It is suprising right? What is the relationship between a fatansy TV show/novel and network science or python(it's not related to a dragon).

If you haven't heard of Game of Thrones, then you must be really good at hiding. Game of Thrones is the hugely popular television series by HBO based on the (also) hugely popular book series A Song of Ice and Fire by George R.R. Martin. In this notebook, we will analyze the co-occurrence network of the characters in the Game of Thrones books. Here, two characters are considered to co-occur if their names appear in the vicinity of 15 words from one another in the books.

![](images/got.png)

Andrew J. Beveridge, an associate professor of mathematics at Macalester College, and Jie Shan, an undergraduate created a network from the book A Storm of Swords by extracting relationships between characters to find out the most important characters in the book(or GoT).

The dataset is publicly avaiable for the 5 books at https://github.com/mathbeveridge/asoiaf. This is an interaction network and were created by connecting two characters whenever their names (or nicknames) appeared within 15 words of one another in one of the books. The edge weight corresponds to the number of interactions. 

Credits:

Blog: https://networkofthrones.wordpress.com

Math Horizons Article: https://www.maa.org/sites/default/files/pdf/Mathhorizons/NetworkofThrones%20%281%29.pdf


```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community
import numpy as np
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

##### Let's load in the datasets


```python
book1 = pd.read_csv('datasets/game_of_thrones_network/asoiaf-book1-edges.csv')
book2 = pd.read_csv('datasets/game_of_thrones_network/asoiaf-book2-edges.csv')
book3 = pd.read_csv('datasets/game_of_thrones_network/asoiaf-book3-edges.csv')
book4 = pd.read_csv('datasets/game_of_thrones_network/asoiaf-book4-edges.csv')
book5 = pd.read_csv('datasets/game_of_thrones_network/asoiaf-book5-edges.csv')
```

The resulting DataFrame book1 has 5 columns: Source, Target, Type, weight, and book. Source and target are the two nodes that are linked by an edge. A network can have directed or undirected edges and in this network all the edges are undirected. The weight attribute of every edge tells us the number of interactions that the characters have had over the book, and the book column tells us the book number.




```python
book1.head()
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
      <th>Source</th>
      <th>Target</th>
      <th>Type</th>
      <th>weight</th>
      <th>book</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Addam-Marbrand</td>
      <td>Jaime-Lannister</td>
      <td>Undirected</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Addam-Marbrand</td>
      <td>Tywin-Lannister</td>
      <td>Undirected</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aegon-I-Targaryen</td>
      <td>Daenerys-Targaryen</td>
      <td>Undirected</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aegon-I-Targaryen</td>
      <td>Eddard-Stark</td>
      <td>Undirected</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aemon-Targaryen-(Maester-Aemon)</td>
      <td>Alliser-Thorne</td>
      <td>Undirected</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Once we have the data loaded as a pandas DataFrame, it's time to create a network. We create a graph for each book. It's possible to create one MultiGraph instead of 5 graphs, but it is easier to play with different graphs.


```python
G_book1 = nx.Graph()
G_book2 = nx.Graph()
G_book3 = nx.Graph()
G_book4 = nx.Graph()
G_book5 = nx.Graph()
```

Let's populate the graph with edges from the pandas DataFrame.


```python
for row in book1.iterrows():
    G_book1.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
```


```python
for row in book2.iterrows():
    G_book2.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
for row in book3.iterrows():
    G_book3.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
for row in book4.iterrows():
    G_book4.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
for row in book5.iterrows():
    G_book5.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'], book=row[1]['book'])
```


```python
books = [G_book1, G_book2, G_book3, G_book4, G_book5]
```

Let's have a look at these edges.


```python
list(G_book1.edges(data=True))[16]
```




    ('Jaime-Lannister', 'Loras-Tyrell', {'book': 1, 'weight': 3})




```python
list(G_book1.edges(data=True))[400]
```




    ('Benjen-Stark', 'Theon-Greyjoy', {'book': 1, 'weight': 4})



### Finding the most important node i.e character in these networks.

Is it Jon Snow, Tyrion, Daenerys, or someone else? Let's see! Network Science offers us many different metrics to measure the importance of a node in a network as we saw in the first part of the tutorial. Note that there is no "correct" way of calculating the most important node in a network, every metric has a different meaning.

First, let's measure the importance of a node in a network by looking at the number of neighbors it has, that is, the number of nodes it is connected to. For example, an influential account on Twitter, where the follower-followee relationship forms the network, is an account which has a high number of followers. This measure of importance is called degree centrality.

Using this measure, let's extract the top ten important characters from the first book (book[0]) and the fifth book (book[4]).


```python
deg_cen_book1 = nx.degree_centrality(books[0])
```


```python
deg_cen_book5 = nx.degree_centrality(books[4])
```


```python
sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:10]
```




    [('Eddard-Stark', 0.3548387096774194),
     ('Robert-Baratheon', 0.2688172043010753),
     ('Tyrion-Lannister', 0.24731182795698928),
     ('Catelyn-Stark', 0.23118279569892475),
     ('Jon-Snow', 0.19892473118279572),
     ('Robb-Stark', 0.18817204301075272),
     ('Sansa-Stark', 0.18817204301075272),
     ('Bran-Stark', 0.17204301075268819),
     ('Cersei-Lannister', 0.16129032258064518),
     ('Joffrey-Baratheon', 0.16129032258064518)]




```python
sorted(deg_cen_book5.items(), key=lambda x:x[1], reverse=True)[0:10]
```




    [('Jon-Snow', 0.1962025316455696),
     ('Daenerys-Targaryen', 0.18354430379746836),
     ('Stannis-Baratheon', 0.14873417721518986),
     ('Tyrion-Lannister', 0.10443037974683544),
     ('Theon-Greyjoy', 0.10443037974683544),
     ('Cersei-Lannister', 0.08860759493670886),
     ('Barristan-Selmy', 0.07911392405063292),
     ('Hizdahr-zo-Loraq', 0.06962025316455696),
     ('Asha-Greyjoy', 0.056962025316455694),
     ('Melisandre', 0.05379746835443038)]




```python
# Plot a histogram of degree centrality
plt.hist(list(nx.degree_centrality(G_book4).values()))
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_22_0.png)



```python
d = {}
for i, j in dict(nx.degree(G_book4)).items():
    if j in d:
        d[j] += 1
    else:
        d[j] = 1
x = np.log2(list((d.keys())))
y = np.log2(list(d.values()))
plt.scatter(x, y, alpha=0.9)
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_23_0.png)


### Exercise

Create a new centrality measure, weighted_degree(Graph, weight) which takes in Graph and the weight attribute and returns a weighted degree dictionary. Weighted degree is calculated by summing the weight of the all edges of a node and find the top five characters according to this measure.


```python
def weighted_degree(G, weight):
    result = dict()
    for node in G.nodes():
        weight_degree = 0
        for n in G.edges([node], data=True):
            weight_degree += n[2]['weight']
        result[node] = weight_degree
    return result
```


```python
plt.hist(list(weighted_degree(G_book1, 'weight').values()))
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_26_0.png)



```python
sorted(weighted_degree(G_book1, 'weight').items(), key=lambda x:x[1], reverse=True)[0:10]
```




    [('Eddard-Stark', 1284),
     ('Robert-Baratheon', 941),
     ('Jon-Snow', 784),
     ('Tyrion-Lannister', 650),
     ('Sansa-Stark', 545),
     ('Bran-Stark', 531),
     ('Catelyn-Stark', 520),
     ('Robb-Stark', 516),
     ('Daenerys-Targaryen', 443),
     ('Arya-Stark', 430)]



### Let's do this for Betweeness centrality and check if this makes any difference

Haha, evil laugh


```python
# First check unweighted, just the structure

sorted(nx.betweenness_centrality(G_book1).items(), key=lambda x:x[1], reverse=True)[0:10]
```




    [('Eddard-Stark', 0.2696038913836117),
     ('Robert-Baratheon', 0.21403028397371796),
     ('Tyrion-Lannister', 0.1902124972697492),
     ('Jon-Snow', 0.17158135899829566),
     ('Catelyn-Stark', 0.1513952715347627),
     ('Daenerys-Targaryen', 0.08627015537511595),
     ('Robb-Stark', 0.07298399629664767),
     ('Drogo', 0.06481224290874964),
     ('Bran-Stark', 0.05579958811784442),
     ('Sansa-Stark', 0.03714483664326785)]




```python
# Let's care about interactions now

sorted(nx.betweenness_centrality(G_book1, weight='weight').items(), key=lambda x:x[1], reverse=True)[0:10]
```




    [('Robert-Baratheon', 0.23341885664466297),
     ('Eddard-Stark', 0.18703429235687297),
     ('Tyrion-Lannister', 0.15311225972516293),
     ('Robb-Stark', 0.1024018949825402),
     ('Catelyn-Stark', 0.10169012330302643),
     ('Jon-Snow', 0.09027684366394043),
     ('Jaime-Lannister', 0.07745109164464009),
     ('Rodrik-Cassel', 0.07667992877670296),
     ('Drogo', 0.06894355184677767),
     ('Jorah-Mormont', 0.0627085149665795)]



#### PageRank
The billion dollar algorithm, PageRank works by counting the number and quality of links to a page to determine a rough estimate of how important the website is. The underlying assumption is that more important websites are likely to receive more links from other websites.


```python
# by default weight attribute in pagerank is weight, so we use weight=None to find the unweighted results
sorted(nx.pagerank_numpy(G_book1, weight=None).items(), key=lambda x:x[1], reverse=True)[0:10]
```




    [('Eddard-Stark', 0.04552079222830669),
     ('Tyrion-Lannister', 0.03301362462493269),
     ('Catelyn-Stark', 0.030193105286631904),
     ('Robert-Baratheon', 0.029834742227736685),
     ('Jon-Snow', 0.02683449952206627),
     ('Robb-Stark', 0.021562941297247527),
     ('Sansa-Stark', 0.02000803404286463),
     ('Bran-Stark', 0.019945786786238345),
     ('Jaime-Lannister', 0.017507847202846896),
     ('Cersei-Lannister', 0.017082604584758083)]




```python
sorted(nx.pagerank_numpy(G_book1, weight='weight').items(), key=lambda x:x[1], reverse=True)[0:10]
```




    [('Eddard-Stark', 0.07239401100498269),
     ('Robert-Baratheon', 0.04851727570509951),
     ('Jon-Snow', 0.047706890624749025),
     ('Tyrion-Lannister', 0.043674378927063114),
     ('Catelyn-Stark', 0.034667034701307456),
     ('Bran-Stark', 0.029774200539800212),
     ('Robb-Stark', 0.029216183645196906),
     ('Daenerys-Targaryen', 0.02708962251302111),
     ('Sansa-Stark', 0.026961778915683174),
     ('Cersei-Lannister', 0.021631679397419022)]



### Is there a correlation between these techniques?

#### Exercise

Find the correlation between these four techniques.

- pagerank
- betweenness_centrality
- weighted_degree
- degree centrality


```python
cor = pd.DataFrame.from_records([nx.pagerank_numpy(G_book1, weight='weight'), nx.betweenness_centrality(G_book1, weight='weight'), weighted_degree(G_book1, 'weight'), nx.degree_centrality(G_book1)])
```


```python
# cor.T
```


```python
cor.T.corr()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.870214</td>
      <td>0.992166</td>
      <td>0.949307</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.870214</td>
      <td>1.000000</td>
      <td>0.857222</td>
      <td>0.871385</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.992166</td>
      <td>0.857222</td>
      <td>1.000000</td>
      <td>0.955060</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.949307</td>
      <td>0.871385</td>
      <td>0.955060</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Evolution of importance of characters over the books

According to degree centrality the most important character in the first book is Eddard Stark but he is not even in the top 10 of the fifth book. The importance changes over the course of five books, because you know stuff happens ;)

Let's look at the evolution of degree centrality of a couple of characters like Eddard Stark, Jon Snow, Tyrion which showed up in the top 10 of degree centrality in first book.

We create a dataframe with character columns and index as books where every entry is the degree centrality of the character in that particular book and plot the evolution of degree centrality Eddard Stark, Jon Snow and Tyrion.
We can see that the importance of Eddard Stark in the network dies off and with Jon Snow there is a drop in the fourth book but a sudden rise in the fifth book


```python
evol = [nx.degree_centrality(book) for book in books]
evol_df = pd.DataFrame.from_records(evol).fillna(0)
evol_df[['Eddard-Stark', 'Tyrion-Lannister', 'Jon-Snow']].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116e37630>




![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_40_1.png)



```python
set_of_char = set()
for i in range(5):
    set_of_char |= set(list(evol_df.T[i].sort_values(ascending=False)[0:5].index))
set_of_char
```




    {'Arya-Stark',
     'Brienne-of-Tarth',
     'Catelyn-Stark',
     'Cersei-Lannister',
     'Daenerys-Targaryen',
     'Eddard-Stark',
     'Jaime-Lannister',
     'Joffrey-Baratheon',
     'Jon-Snow',
     'Margaery-Tyrell',
     'Robb-Stark',
     'Robert-Baratheon',
     'Sansa-Stark',
     'Stannis-Baratheon',
     'Theon-Greyjoy',
     'Tyrion-Lannister'}



##### Exercise

Plot the evolution of weighted degree centrality of the above mentioned characters over the 5 books, and repeat the same exercise for betweenness centrality.


```python
evol_df[list(set_of_char)].plot(figsize=(29,15))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x117b28400>




![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_43_1.png)



```python
evol = [nx.betweenness_centrality(graph, weight='weight') for graph in [G_book1, G_book2, G_book3, G_book4, G_book5]]
evol_df = pd.DataFrame.from_records(evol).fillna(0)

set_of_char = set()
for i in range(5):
    set_of_char |= set(list(evol_df.T[i].sort_values(ascending=False)[0:5].index))


evol_df[list(set_of_char)].plot(figsize=(19,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116e4a4a8>




![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_44_1.png)


### So what's up with  Stannis Baratheon?


```python
nx.draw(nx.barbell_graph(5, 1), with_labels=True)
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_46_0.png)



```python
sorted(nx.degree_centrality(G_book5).items(), key=lambda x:x[1], reverse=True)[:5]
```




    [('Jon-Snow', 0.1962025316455696),
     ('Daenerys-Targaryen', 0.18354430379746836),
     ('Stannis-Baratheon', 0.14873417721518986),
     ('Tyrion-Lannister', 0.10443037974683544),
     ('Theon-Greyjoy', 0.10443037974683544)]




```python
sorted(nx.betweenness_centrality(G_book5).items(), key=lambda x:x[1], reverse=True)[:5]
```




    [('Stannis-Baratheon', 0.45283060689247934),
     ('Daenerys-Targaryen', 0.2959459062106149),
     ('Jon-Snow', 0.24484873673158666),
     ('Tyrion-Lannister', 0.20961613179551256),
     ('Robert-Baratheon', 0.17716906651536968)]



#### Community detection in Networks
A network is said to have community structure if the nodes of the network can be easily grouped into (potentially overlapping) sets of nodes such that each set of nodes is densely connected internally.

We will use louvain community detection algorithm to find the modules in our graph.


```python
plt.figure(figsize=(15, 15))

partition = community.best_partition(G_book1)
size = float(len(set(partition.values())))
pos = nx.kamada_kawai_layout(G_book1)
count = 0
colors = ['red', 'blue', 'yellow', 'black', 'brown', 'purple', 'green', 'pink']
for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G_book1, pos, list_nodes, node_size = 20,
                                node_color = colors[count])
    count = count + 1



nx.draw_networkx_edges(G_book1, pos, alpha=0.2)
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_50_0.png)



```python
d = {}
for character, par in partition.items():
    if par in d:
        d[par].append(character)
    else:
        d[par] = [character]
d
```




    {0: ['Addam-Marbrand',
      'Jaime-Lannister',
      'Tywin-Lannister',
      'Tyrion-Lannister',
      'Bronn',
      'Chiggen',
      'Marillion',
      'Shae',
      'Shagga',
      'Vardis-Egen',
      'Willis-Wode',
      'Colemon',
      'Chella',
      'Conn',
      'Coratt',
      'Dolf',
      'Gunthor-son-of-Gurn',
      'Harys-Swyft',
      'Kevan-Lannister',
      'Jyck',
      'Morrec',
      'Kurleket',
      'Leo-Lefford',
      'Mord',
      'Timett',
      'Ulf-son-of-Umar'],
     1: ['Aegon-I-Targaryen',
      'Daenerys-Targaryen',
      'Aggo',
      'Drogo',
      'Jhogo',
      'Jorah-Mormont',
      'Quaro',
      'Rakharo',
      'Cohollo',
      'Haggo',
      'Qotho',
      'Doreah',
      'Eroeh',
      'Illyrio-Mopatis',
      'Irri',
      'Jhiqui',
      'Mirri-Maz-Duur',
      'Viserys-Targaryen',
      'Jommo',
      'Ogo',
      'Rhaego',
      'Fogo'],
     2: ['Eddard-Stark',
      'Aerys-II-Targaryen',
      'Brandon-Stark',
      'Gerold-Hightower',
      'Jon-Arryn',
      'Robert-Baratheon',
      'Alyn',
      'Harwin',
      'Jory-Cassel',
      'Tomard',
      'Arthur-Dayne',
      'Cersei-Lannister',
      'Petyr-Baelish',
      'Vayon-Poole',
      'Arys-Oakheart',
      'Balon-Greyjoy',
      'Renly-Baratheon',
      'Barristan-Selmy',
      'Pycelle',
      'Varys',
      'Lyanna-Stark',
      'Cayn',
      'Janos-Slynt',
      'Stannis-Baratheon',
      'Rhaegar-Targaryen',
      'Daryn-Hornwood',
      'Torrhen-Karstark',
      'Gendry',
      'Howland-Reed',
      'Jacks',
      'Joss',
      'Porther',
      'Raymun-Darry',
      'Tobho-Mott',
      'Tregar',
      'Varly',
      'Wyl-(guard)',
      'Wylla',
      'Oswell-Whent',
      'Heward',
      'Hugh',
      'Lancel-Lannister'],
     3: ['Aemon-Targaryen-(Maester-Aemon)',
      'Alliser-Thorne',
      'Bowen-Marsh',
      'Chett',
      'Clydas',
      'Jeor-Mormont',
      'Jon-Snow',
      'Samwell-Tarly',
      'Albett',
      'Halder',
      'Rast',
      'Grenn',
      'Pypar',
      'Benjen-Stark',
      'Yoren',
      'Jaremy-Rykker',
      'Mance-Rayder',
      'Dareon',
      'Donal-Noye',
      'Dywen',
      'Todder',
      'Hobb',
      'Jafer-Flowers',
      'Matthar',
      'Othor',
      'Randyll-Tarly'],
     4: ['Arya-Stark',
      'Desmond',
      'Ilyn-Payne',
      'Jeyne-Poole',
      'Joffrey-Baratheon',
      'Meryn-Trant',
      'Mordane',
      'Mycah',
      'Myrcella-Baratheon',
      'Sandor-Clegane',
      'Sansa-Stark',
      'Syrio-Forel',
      'Tommen-Baratheon',
      'Balon-Swann',
      'Boros-Blount',
      'Beric-Dondarrion',
      'Gregor-Clegane',
      'Loras-Tyrell',
      'Thoros-of-Myr',
      'High-Septon-(fat_one)',
      'Marq-Piper',
      'Mace-Tyrell',
      'Paxter-Redwyne',
      'Maegor-I-Targaryen'],
     5: ['Bran-Stark',
      'Catelyn-Stark',
      'Rickon-Stark',
      'Robb-Stark',
      'Rodrik-Cassel',
      'Luwin',
      'Theon-Greyjoy',
      'Hali',
      'Hallis-Mollen',
      'Hodor',
      'Hullen',
      'Joseth',
      'Nan',
      'Osha',
      'Rickard-Karstark',
      'Rickard-Stark',
      'Stiv',
      'Brynden-Tully',
      'Edmure-Tully',
      'Hoster-Tully',
      'Lysa-Arryn',
      'Nestor-Royce',
      'Walder-Frey',
      'Donnel-Waynwood',
      'Eon-Hunter',
      'Jon-Umber-(Greatjon)',
      'Masha-Heddle',
      'Moreo-Tumitis',
      'Mya-Stone',
      'Mychel-Redfort',
      'Robert-Arryn',
      'Stevron-Frey',
      'Tytos-Blackwood',
      'Wendel-Manderly',
      'Clement-Piper',
      'Karyl-Vance',
      'Galbart-Glover',
      'Roose-Bolton',
      'Maege-Mormont',
      'Jonos-Bracken',
      'Lyn-Corbray'],
     6: ['Waymar-Royce', 'Gared', 'Will-(prologue)'],
     7: ['Danwell-Frey', 'Hosteen-Frey', 'Jared-Frey']}




```python
nx.draw(nx.subgraph(G_book1, d[3]))
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_52_0.png)



```python
nx.draw(nx.subgraph(G_book1, d[1]))
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_53_0.png)



```python
nx.density(G_book1)
```




    0.03933068828704502




```python
nx.density(nx.subgraph(G_book1, d[4]))
```




    0.19927536231884058




```python
nx.density(nx.subgraph(G_book1, d[4]))/nx.density(G_book1)
```




    5.066663488431223



#### Exercise 

Find the most important node in the partitions according to degree centrality of the nodes.


```python
max_d = {}
deg_book1 = nx.degree_centrality(G_book1)

for group in d:
    temp = 0
    for character in d[group]:
        if deg_book1[character] > temp:
            max_d[group] = character
            temp = deg_book1[character]
```


```python
max_d
```




    {0: 'Tyrion-Lannister',
     1: 'Daenerys-Targaryen',
     2: 'Eddard-Stark',
     3: 'Jon-Snow',
     4: 'Sansa-Stark',
     5: 'Catelyn-Stark',
     6: 'Waymar-Royce',
     7: 'Danwell-Frey'}



## A bit about power law in networks


```python
G_random = nx.erdos_renyi_graph(100, 0.1)
```


```python
nx.draw(G_random)
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_62_0.png)



```python
G_ba = nx.barabasi_albert_graph(100, 2)
```


```python
nx.draw(G_ba)
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_64_0.png)



```python
# Plot a histogram of degree centrality
plt.hist(list(nx.degree_centrality(G_random).values()))
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_65_0.png)



```python
plt.hist(list(nx.degree_centrality(G_ba).values()))
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_66_0.png)



```python
G_random = nx.erdos_renyi_graph(2000, 0.2)
G_ba = nx.barabasi_albert_graph(2000, 20)
```


```python
d = {}
for i, j in dict(nx.degree(G_random)).items():
    if j in d:
        d[j] += 1
    else:
        d[j] = 1
x = np.log2(list((d.keys())))
y = np.log2(list(d.values()))
plt.scatter(x, y, alpha=0.9)
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_68_0.png)



```python
d = {}
for i, j in dict(nx.degree(G_ba)).items():
    if j in d:
        d[j] += 1
    else:
        d[j] = 1
x = np.log2(list((d.keys())))
y = np.log2(list(d.values()))
plt.scatter(x, y, alpha=0.9)
plt.show()
```


![png](7-game-of-thrones-case-study-instructor_files/7-game-of-thrones-case-study-instructor_69_0.png)

