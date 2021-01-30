#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 21:32:26 2021

@author: miguel
"""

#%%

import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import matplotlib as mpl
# from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestRegressor as RFR 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from collections import defaultdict

X, y = load_boston(return_X_y=True) 

def rfrcv(target, train_data, msl=None, mss=None):
    if mss:
        val = cross_val_score(
            RFR(min_samples_split=mss, 
                criterion='mse',
                random_state=2,
            ),
            train_data, target, 
            cv=6, n_jobs=7
        ).mean()
        return val
    if msl:
        val = cross_val_score(
            RFR(min_samples_leaf=msl, 
                criterion='mse',
                random_state=2,
            ),
            train_data, target, 
            cv=6, n_jobs=7
        ).mean()
        return val        
 
def lerp(a, b, f):
    return a + f * (b - a)

params = {'min_samples_leaf': (0.01, 0.35),
         'min_samples_split': (0.01, 0.65), 
        }


#%% exploring variation in hyperparameter over sample size
size_grid = list(range(50, len(X), 50)) 
steps = 20

grid_search_mss = dict()
grid_search_msl = dict()
for size in size_grid:
    print(size, len(X))
    grid_search_mss[size] = list() 
    grid_search_msl[size] = list() 
    
    idx = np.random.randint(len(X), size=size)
    train_data = X[idx,:]
    target = y[idx] 
     
    for n1 in range(steps): 
        mss = round(lerp(params['min_samples_split'][0], params['min_samples_split'][1], n1/(steps-1)), 3) 
        val = rfrcv(target=target, train_data=train_data, mss=mss)   
        grid_search_mss[size].append([val, mss,]) 

    for n2 in range(steps): 
        msl = round(lerp(params['min_samples_leaf'][0], params['min_samples_leaf'][1], n2/(steps-1)), 3) 
        val = rfrcv(target=target, train_data=train_data, msl=msl)   
        grid_search_msl[size].append([val, msl,]) 
        
#%%
 
fig, (ax1, ax2) = plt.subplots(1, 2)
for i, size in enumerate(size_grid):
    colormap = mpl.cm.autumn 
    colorst = [colormap(25*i) for i in range(len(size_grid))] 
    
    d1 = grid_search_mss[size]  
    score1, param1 = list(map(list, zip(*d1)))
    ax1.plot(param1, score1, alpha=0.7, label=size, c=colorst[i])
    ax1.set_xlabel('min_samples_split') 
    # ax1.colorbar()
    # ax1.legend(fontsize='small')
    
    d2 = grid_search_msl[size]     
    score2, param2 = list(map(list, zip(*d2)))     
    ax2.plot(param2, score2, alpha=0.7, label=size, c=colorst[i])
    ax2.set_xlabel('min_samples_leaf')
    # ax2.legend(fontsize='small') 


    cmap = plt.cm.rainbow
    norm = mpl.colors.Normalize(vmin=5, vmax=95)
    
    # fig, ax = plt.subplots()
    # ax.bar(df.x, df.y, color=cmap(norm(df.c.values)))
    # ax.set_xticks(df.x)
    
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    # sm.set_array([])  # only needed for matplotlib < 3.1
plt.colorbar(sm)


'''
There is a significant amount of overlap for all sample sizes.
This indicates that to some extent, experiments with very small samples might
be informative for the optimal parameter.
'''    
#%% exploring variance in results from sample sizes

size_grid = list(range(50, len(X), 100))  
repeats = 100 # amount of models to estimate variation caused by different samples
steps = 3

variance = defaultdict(list) 
for it in range(repeats):
    print(it)
    
    for size in size_grid: 
        lst = list()
        idx = np.random.randint(len(X), size=size)
        train_data = X[idx,:]
        target = y[idx] 
         
        for n1 in range(steps): 
            mss = round(lerp(params['min_samples_split'][0], params['min_samples_split'][1], n1/(steps-1)), 3) 
            val = rfrcv(target=target, train_data=train_data, mss=mss)
            lst.append([val, mss,])
        variance[size].append(lst) 
 
 
#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3) 
colormap = mpl.cm.autumn 
# colormap = plt.cm.nipy_spectral #nipy_spectral, Set1,Paired   #gist_ncar
colorst = [colormap(40*i) for i in range(len(size_grid))]

for j in range(len(variance[50])):
    score, param = list(map(list, zip(*variance[50][j])))  
    ax1.plot(param, score, alpha=0.2, c=colorst[1]) 
    # ax1.set_xlabel('min_samples_split')
    # ax1.title.set_text('10%')

    score, param = list(map(list, zip(*variance[150][j])))  
    ax2.plot(param, score, alpha=0.2, c=colorst[3]) 
    # ax2.set_xlabel('min_samples_split')
    # ax2.title.set_text('30%')
    
    score, param = list(map(list, zip(*variance[450][j])))  
    ax3.plot(param, score, alpha=0.2, c=colorst[4]) 
    # ax3.set_xlabel('min_samples_split')
    # ax3.title.set_text('90%')
 

#%%
statistics = defaultdict(lambda: defaultdict(list))
for i, size in enumerate(size_grid):
    d = variance[size] 
    for i in d:
        for j in range(len(i)): 
            statistics[size][i[j][1]].append(i[j][0])
    
    for k1 in statistics:
        for k2 in statistics[k1]:
            print(np.mean(statistics[k1][k2]))
            print(np.std(statistics[k1][k2]))

means, sds = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
for i, size in enumerate(size_grid):
    for k1 in statistics:
        for k2 in statistics[k1]:
            means[k1][k2] = np.mean(statistics[k1][k2])
            sds[k1][k2] = np.std(statistics[k1][k2])
 
#%%

fig, (ax1, ax2, ax3) = plt.subplots(1, 3) 
colormap = mpl.cm.autumn  
colorst = [colormap(40*i) for i in range(len(size_grid))]
 

size=50
x = [k for k in means[size]]
mu = [means[size][k] for k in means[size]]
mu_2sd = [means[size][k] +  sds[size][k] for k in sds[size]]
mu__2sd = [means[size][k] -  sds[size][k] for k in sds[size]]

ax1.plot(x, mu, c=colorst[1], lw=2)
# ax1.plot(x, mu, c=colorst[1], lw=10, alpha=0.3)
ax1.plot(x, mu_2sd, c=colorst[1], alpha=0.2)
ax1.plot(x, mu__2sd, c=colorst[1], alpha=0.2)
ax1.fill_between(x, mu__2sd, mu_2sd, color="grey", alpha=0.2)

size=150
x = [k for k in means[size]]
mu = [means[size][k] for k in means[size]]
mu_2sd = [means[size][k] +  sds[size][k] for k in sds[size]]
mu__2sd = [means[size][k] -   sds[size][k] for k in sds[size]]

ax2.plot(x, mu, c=colorst[3], lw=2)
# ax2.plot(x, mu, c=colorst[3], lw=10, alpha=0.3)
ax2.plot(x, mu_2sd, c=colorst[3], alpha=0.2)
ax2.plot(x, mu__2sd, c=colorst[3], alpha=0.2)
ax2.fill_between(x, mu__2sd, mu_2sd, color="grey", alpha=0.2)
ax2.set_xlabel('min_samples_split')

size=450
x = [k for k in means[size]]
mu = [means[size][k] for k in means[size]]
mu_2sd = [means[size][k] +   sds[size][k] for k in sds[size]]
mu__2sd = [means[size][k] -  sds[size][k] for k in sds[size]]

ax3.plot(x, mu, c=colorst[4], lw=2)
# ax3.plot(x, mu, c=colorst[4], lw=10, alpha=0.3)
ax3.plot(x, mu_2sd, c=colorst[4], alpha=0.2)
ax3.plot(x, mu__2sd, c=colorst[4], alpha=0.2)
ax3.fill_between(x, mu__2sd, mu_2sd, color="grey", alpha=0.2)
 
#%%
for size in size_grid:
    score, param = list(map(list, zip(*variance[size]))) 
    plt.scatter(param, score, alpha=0.5) 
     
'''
As expected, smaller samples have greater variance than larger samples.
Larger samples will have a very significant overlap with the population which
means that there's less to vary in the training data, and the results should be 
similar.
'''

#%% comparing mean/median of scores from smaller and biggest size. Do they match?

param_median = defaultdict(list)
param_mean = defaultdict(list)

for size in size_grid:
    parameter_set = set(sorted([p for s, p in variance[size][0]]))
    for p1 in parameter_set: 
        s_lst = [s for s, p in variance[size] if p1 == p]
        param_median[p1].append(round(np.median(s_lst),3))
        param_mean[p1].append(round(np.mean(s_lst),3))
        # print(f'{size}: {p1}   {np.median(s_lst)}


# the_table = plt.table(rowLabels=param_median.values,
#                       rowColours=colors,
#                       colLabels=columns,
#                       loc='bottom')
'''
An analysis of the plots above seems to indicate that the distribution
overall seems to resemble and inform on the populations true value for the optimal parameters
The median needs to be used because in small samples the variance is so large that
outliers are frequently generated and perturb the mean.
'''

#%% focusing on small samples distributions for all parameter values: gaussian or not?
# from scipy.stats import shapiro#, normaltest



parameter_set = set(sorted([p for s, p in variance[150][0]]))
for p1 in parameter_set: 
    
    for size in size_grid:
        # s_lst = list()
        all_sizes = list()
        for line in variance[size]: 
            s_lst = [s for s, p in line if p1 == p] 
            # s, p = shapiro(s_lst)
            # if p > 0.05:
            #     print(f'{p1} {size} distribution is likely gaussian')
            # else:
            #     print(f'{p1} {size} distribution is likely not gaussian')
            all_sizes.append(s_lst[0])
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.tick_params(left=False) 
        ax.set(yticklabels=[])
        ax.set_title(f'min_samples_split:{p1} size:{size}' )
         
        ax.hist(all_sizes, bins='auto', alpha=0.5, color=colorst[2])
        ax.set_xlim(0,1)
 
        plt.show()

 
'''
The distributions seem to be somewhat symmetric, but a normality test would be needed.
If these distributions are normal, we can infer that the mean of these distributions 
could be seen as the mean of the true population - CLT corollary.
The distributions seem to share some sort of central tendency
'''
#%% testing normality
 