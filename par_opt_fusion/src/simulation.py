# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 08:22:10 2021

@author: z003njns
"""

#%%

import matplotlib.pyplot as plt
import numpy as np
from math import sin

def sample_size_to_noise(sample_size):
    return np.random.normal(0.0, 1 - sample_size, 1)
    
def sample_signal(sample_size, x):
    
    noise = sample_size_to_noise(sample_size)
    
    signal = 2*x + sin(10*x) + sin(50*x) + noise
    
    return float(signal)


#%%
    
xx = [0.01*i for i in range(100)]
 
for s in range(100):    
    yy = list()
    
    sample_size=0.9
    for x in xx:
        yy.append(sample_signal(sample_size, x))

    plt.plot(xx, yy, alpha=0.01)
    # plt.ylim(-4,4)


#%%
execution_cost = dict()

def complexity(n): 
    return n**2

for size in range(20, 101, 1):
    execution_cost[size] = complexity(size) 

#%%
from collections import defaultdict


samples = defaultdict(list)
for sample_size in [0.1, 0.25, 0.5, 0.9]:
    xx = [0.01*i for i in range(0, 100, 10)]
    for x in xx:
        samples[sample_size].append(sample_signal(sample_size, x))
     
    plt.plot(xx, samples[sample_size], lw=sample_size)
    # plt.legend(''.join(str(sample_size)))    
