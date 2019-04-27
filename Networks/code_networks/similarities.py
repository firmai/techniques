"""
Exercises for similarities/distances
"""
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.stats as stats
import matplotlib.pyplot as plt
import dzcnapy_plotlib as dzcnapy

# Hamming distance

data = {
    "potato" : {"shape": "round", "color": "yellowish", "starchy": True},
    "carrot" : {"shape": "conic", "color": "orange", "starchy": False},
    "corn" : {"shape": "conic", "color": "yellowish", "starchy": True},
    "turnip" : {"shape": "round", "color": "yellowish", "starchy": False}
    }
atts = set.union(*[set(x.keys()) for x in data.values()])

sim_2dlist = [[sum(data[v1][att] == data[v2][att] for att in atts) \
                   / len(atts) for v1 in data] for v2 in data]

sim_2dlist = [[1 - dist.hamming(list(data[v1].values()),
                                list(data[v2].values())) for v1 in data]
              for v2 in data]

sim_array = np.array(sim_2dlist)
#array([[ 1.        ,  0.33333333,  0.        ,  0.33333333],
#       [ 0.33333333,  1.        ,  0.66666667,  0.33333333],
#       [ 0.        ,  0.66666667,  1.        ,  0.66666667],
#       [ 0.33333333,  0.33333333,  0.66666667,  1.        ]])

sim_dataframe = pd.DataFrame(sim_array, columns=data, index=data)
#          carrot      corn    potato    turnip
#carrot  1.000000  0.333333  0.000000  0.333333
#corn    0.333333  1.000000  0.666667  0.333333
#potato  0.000000  0.666667  1.000000  0.666667
#turnip  0.333333  0.333333  0.666667  1.000000

# Manhattan distance
hwdata = [[65.78, 112.99],
          [71.52, 136.49],
          [69.40, 153.03],
          [68.22, 142.34],
          [67.79, 144.30]]
hw_array = np.array(hwdata)
five_guys = np.array([[dist.cityblock(x, y) for x in hw_array]
                      for y in hw_array])

# Normalized
hw_range = hw_array.max(axis=0) - hw_array.min(axis=0)
hw_norm = (hw_array - hw_array.min(axis=0)) / hw_range
five_guys_norm = np.array([[dist.cityblock(x, y) for x in hw_array]
                           for y in hw_array])

# Cosine distance
# Hrs per year @ [7,12) mph
#https://www.meteoblue.com/en/weather/forecast/modelclimate/
winds = {
    "Anchorage": (58,60,132,552,291,180,88,62,58,36,20,4,3,16,119,81),
    "Boston": (93,104,106,101,80,82,82,110,216,292,281,205,246,204,159,86),
    "Chicago": (115,195,122,109,86,120,157,210,273,196,139,101,113,106,
                107,115),
    "San Francisco": (35,67,156,616,1208,894,268,67,2,0,0,0,2,9,22,35)
    }

# Draw the wind roses
angles = np.linspace(0, 2 * np.pi, 16)
ax = plt.subplot(111, polar=True)
for name, wind in winds.items():
    ax.plot(angles, wind, "-o", label=name)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.legend(loc=2)
dzcnapy.plot("windroses")

wind_cities_cosine = pd.DataFrame({y: [1 - dist.cosine(winds[x], winds[y]) 
                                       for x in winds] for y in winds},
                                  index=winds.keys())

# Pearson correlation
wind_cities_pearson = pd.DataFrame({y: [stats.pearsonr(winds[x], 
                                                       winds[y])[0] 
                                        for x in winds] for y in winds}, 
                                   index=winds.keys())
pd.DataFrame(winds).corr()
