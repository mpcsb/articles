# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:18:14 2021

@author: z003njns
"""
#%%
from numpy.random import laplace as lap
from numpy import std, mean
import matplotlib.pyplot as plt
from random import random


size = 10200
d1 = [round(lap(0, 1), 1) for _ in range(size)]
d2 = [round(lap(6, 1), 1) for _ in range(size)]
population = list(d1 + d2)
average = round(sum(population)/len(population), 3)
sd_pop = round(std(population), 3)

plt.hist(population, bins=30, alpha=0.4, color='red')
plt.title(str(average) +',   ' + str(sd_pop))
plt.show()

means = []
sds = []
frac = 0.005
for i in range(10): #10 samples
    sample = [population[i] for i in range(len(population)) if random() < frac]
    mu = round(sum(sample)/len(sample),3)

    means.append(mu)
    sd = std(sample)
    sds.append(sd)
# sd = round(std(sample), 3)

plt.hist(sample, bins=10)
plt.title(str(mu))
plt.show()

print(average, sd_pop)
print(mean(means), mean(sds))