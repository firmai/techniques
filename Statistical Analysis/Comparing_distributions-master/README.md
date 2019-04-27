COMPARE DISTRIBUTIONS V1.1
=======================

It is a simple tool to compare two normal, Poisson or Gamma distributions.
Given the means and standard deviations of the distributions it will give
a likelihood that a random draw from a population with a higher mean
is bigger than a random draw from a population with the smaller mean.

Experimental results often report statistically significant differences
that are not very meaningful in every day life, i.e. their actual impact
is very small. With big enough sample we can often detect differences,
which are very small.

Let's say that some researcher reported that with p-value of 0.001 an
African swallow can lift a heavier coconut than a European swallow. This
assertion is based on results of an experiment carried on a population
of 1,000,000 birds. The researcher also provided information about the
experimental results saying that an African swallow on average could
carry a coconut weighting 478g (SD=189g), while the European only a
coconut weighting 477g (SD=156g). The numbers already look small, but
how likely it is that if we took one random African and one random
European swallow, the African one could carry a heavier coconut than the
European one? (The answer is about 50.11%).

This simple tool calculates an answer to this question and helps in
interpreting experimental results.

The output of the script is a result of the test and a graph illustrating
the two distributions.

Installation
------------

The script has been tested with Python 2.7 (Anaconda Python Distribution - Windows 8.1 64-bit)

Modules required:
- numpy
- matplotlib
