print """
COMPARE DISTRIBUTIONS V1.1

It is a simple tool to compare two normal, Poisson or Gamma distributions.
Given the means and standard deviations of the distributions it will give
a likelihood that a random draw from a population with a higher mean
is bigger than a random draw from a population with the smaller mean.

The output of the script is a result of the test and a graph illustrating
the two distributions.

ver. 1.0 2011
ver. 1.1 2014

Maciej Workiewicz 2014

"""

import numpy as np
from matplotlib import pyplot as plt

i = 1000000  # change to increase/decrease precision and running time

_dist = int(raw_input('''What type of distributions do you want to compare?
            1 - Normal
            2 - Poisson
            3 - Gamma
            '''))

H_OUT = np.zeros(i)
S_OUT = np.zeros(i)
OUT_X = np.zeros(i)
if _dist == 1:
    h_mean = float(raw_input("Mean of the higher distribution (H): "))
    h_std = float(raw_input("SD of the higher distribution (H): "))
    s_mean = float(raw_input("Mean of the smaller distribution (S): "))
    s_std = float(raw_input("SD of the smaller distribution (S): "))
    H_name = ", H ~N(" + str(h_mean) + ", " + str(h_std) + ")"
    S_name = ", S ~N(" + str(s_mean) + ", " + str(s_std) + ")"

    for x in np.arange(i):
        _h = np.random.normal(h_mean, h_std)
        _s = np.random.normal(s_mean, s_std)
        if _h > _s:
            OUT_X[x] = 1
        else:
            OUT_X[x] = 0
        H_OUT[x] = _h
        S_OUT[x] = _s
elif _dist == 2:
    h_lambda = float(raw_input("Lambda of the higher distribution (H): "))
    s_lambda = float(raw_input("Lambda of the smaller distribution (S): "))
    H_name = ", H ~P(" + str(h_lambda) + ")"
    S_name = ", S ~P(" + str(s_lambda) + ")"
    for x in np.arange(i):
        _h = np.random.poisson(h_lambda)
        _s = np.random.poisson(s_lambda)
        if _h > _s:
            OUT_X[x] = 1
        else:
            OUT_X[x] = 0
        H_OUT[x] = _h
        S_OUT[x] = _s

elif _dist == 3:
    h_gamma = float(raw_input("Gamma of the higher distribution (H): "))
    h_theta = float(raw_input("Theta of the higher distribution (H): "))
    s_gamma = float(raw_input("Gamma of the smaller distribution (S): "))
    s_theta = float(raw_input("Theta of the smaller distribution (S): "))
    H_name = ", H ~Gamma(" + str(h_gamma) + ", " + str(h_theta) + ")"
    S_name = ", S ~Gamma(" + str(s_gamma) + ", " + str(s_theta) + ")"

    for x in np.arange(i):
        _h = np.random.gamma(h_gamma, h_theta)
        _s = np.random.gamma(s_gamma, s_theta)
        if _h > _s:
            OUT_X[x] = 1
        else:
            OUT_X[x] = 0
            H_OUT[x] = _h
            S_OUT[x] = _s

AVG_X = np.mean(OUT_X)
print("The likelihood of H>S is: " + str(AVG_X))

# Now the histogram
plt.figure()
plt.hist(H_OUT, normed=True, histtype='stepfilled', color='b', alpha=0.5,
         label='H', bins=50)
plt.hist(S_OUT, normed=True, histtype='stepfilled', color='r', alpha=0.5,
         label='S', bins=50)
plt.legend()
plt.title("Comparison," + H_name + S_name)
plt.ylabel("Probability")
plt.xlabel("Value")
plt.show()

# end of line
