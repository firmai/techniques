# This file produces all network type figures
import networkx as nx
import calendar
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import dzcnapy_plotlib as dzcnapy

F = nx.Graph()

for i in range(1, 9):
    for j,k in zip('ABCDEFG','BCDEFGH'):
        F.add_edge('{}{}'.format(j,i),'{}{}'.format(k,i))

for i in 'ABCDEFGH':
    for j,k in zip(range(1,8), range(2,9)):
        F.add_edge('{}{}'.format(i,j), '{}{}'.format(i,k))

pos = graphviz_layout(F)
nx.draw_networkx(F, pos, **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("checkerboard")

#Tree
lincoln_list = [
    ("A.L.", "Edward Baker L."), ("A.L.", "Robert Todd L."), 
    ("A.L.", "William Wallace L."), ("A.L.", "Thomas L. III"),
    ("Jessie Harlan L.", "Mary L. Beckwith"), 
    ("Jessie Harlan L.", "Robert Todd L. Beckwith"), 
    ("Mary L.", "L. Isham"), ("Robert Todd L.", "A.L. II"), 
    ("Robert Todd L.", "Jessie Harlan L."), 
    ("Robert Todd L.", "Mary L."), ("Thomas L.", "A.L."), 
    ("Thomas L.", "Sarah L. Grigsby"), ("Thomas L.", "Thomas L. Jr."),
]
F = nx.DiGraph(lincoln_list)

pos = graphviz_layout(F)
nx.draw_networkx(F, pos, **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("lincoln")

# Linear
dates = ("Born 1809", "Married 1842", "Elected Representative 1847", 
         "Elected President 1861", "Died 1865")
F.add_edges_from(zip(dates, dates[1:]))

# Ring
F.add_edges_from(zip(calendar.month_name[1:], 
                     calendar.month_name[2:] + calendar.month_name[1:2]))

pos = graphviz_layout(F)
nx.draw_networkx(F, pos, **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("types-of-networks")
