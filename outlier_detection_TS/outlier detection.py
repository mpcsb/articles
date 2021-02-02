#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identifying outliers in short time series

@author: miguelcbatista@gmail.com
"""

'''
The key idea behind gaussian processes regression is the ability to create a 
data generating process without much data. 
The cubic computational complexity makes the algorithm unfit for large volume
of data, which in a sense restricts the range of problems it can tackle.

One fairly frequent case would be monthly time series. Say, total volume of sales
or expenses associated with business.
A typical scenario would be composed of a collection of tuples (year-month, target amount)
ordered in time.

If our focus is truely to understand the past and not on forecasting, we can expand 
the scope of our data and use exogenous variables (like interest rates, prices for materials).
This would be data that we cannot access in future, so our model cannot be based in
it.
So, a series could be something like (year-month, signal 1, ..., signal n, target mount)
If signal i justifies why the target is varying, than it's not an outlier.
If nothing in the data justifies a data point, then it's an outlier.

A few years of data contains tens of data points, which is totally reasonable and
within the bounds for gaussian processes.

'''


'''
One common request related to time series is to identify outliers.
Understanding that a point in time is not following the probabilistic model
that you think makes sense for that series, is an indicator that either something
happened or that the model itself is not complete (duh) and can be improved.
'''

'''
Essentially, for gaussian processes we want to discover which of those points
is considered extraordinary and is not solely caused by noise, but to some other
exogenous effect.
'''


'''
Post scenarios:
    1) Simple series. Basic processing of data. Introduce periodic kernel and changepoints.
    2) Forecasting - essentially illustrates the process. Taking mean and variance from samples.
    3) Looping series, leave one out. The gap will be described by algorithm. How to model with this gap?
'''

#%% Setting up a scenario
import random
import itertools as it
import math 
import numpy as np 
import matplotlib.pyplot as plt
import logging 


import pymc3 as pm
import arviz as az
from pymc3.gp.util import plot_gp_dist


import warnings
warnings.filterwarnings("ignore")

#%%
# for n in [ 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
years = 2
noise_magnitude = 3
period = [2,1,5,1,1,0,1,1,2,2,3,3] * years
trend = [math.sin(i) + i for i in range(12 * years)]

noise = [(random.random() - 0.5) * noise_magnitude for _ in range(12 * years)]
 

def join_components(mode, *components): 
    signal = [0] * len(components[0])
    for c in components:  
        for i, p in enumerate(c): 
            if mode == 'add':
                signal[i] += p
            if mode == 'mult':
                signal[i] *= p
                
    return np.array(signal)

series = join_components('add', period, trend, noise) 
 

logger = logging.getLogger("pymc3")
logger.propagate = False

# X = np.linspace(0, len(series), len(series))[:,None]
X = np.array([i for i in range(len(series))])[:,None]

with pm.Model() as ts_gp:
    coef = pm.HalfNormal('coef', sigma=0.5) 
    lin_func = pm.gp.mean.Linear(coeffs=coef, intercept=0.0)
    
    ls_matern = pm.HalfNormal('ls_matern', sigma=0.5)
    cov = pm.gp.cov.Matern52(1, ls=ls_matern) # base for additive gp
    ls_period = pm.HalfNormal('ls_period', sigma=0.5)
    cov += pm.gp.cov.Periodic(1, period=12, ls=ls_period)     
    
    sigma = pm.HalfCauchy("sigma", beta=1)      
    
    gp = pm.gp.Marginal(mean_func=lin_func, cov_func=cov) 
    gp.marginal_likelihood("y", X=X, y=series, noise=sigma) 
    
    trace = pm.sample(5_000, tune=2_000, chains=1, progressbar=True)

 
az.plot_trace(trace)  

forecasted_years = 0
n_new = 12 * forecasted_years
X_new = np.array(list(range(0, len(series) + n_new)))[:,None]
with ts_gp:
        f_pred = gp.conditional("f_pred", X_new)
        pred_samples = pm.sample_posterior_predictive(trace, 
                            vars=[f_pred], samples=2_000)

fig = plt.figure(figsize=(8,6))
ax = fig.gca()    
plot_gp_dist(ax, pred_samples["f_pred"], X_new); 
plt.plot(X, series, 'ok', ms=5, alpha=0.6, label="Observations") 
plt.axvline(len(series)) 
plt.title(f"Posterior distribution @ noise={noise_magnitude}"); plt.legend();
plt.show()

# trace_df = pm.backends.tracetab.trace_to_dataframe(trace)
#%%

pred = pred_samples['f_pred'] 
a = pred[:,1]
np.percentile(pred[:,1], 89)

mean = np.array([np.mean(pred[:,i]) for i in range(len(X_new))])
sd = np.array([np.std(pred[:,i]) for i in range(len(X_new))])

 
fig = plt.figure(figsize=(12,5)); #ax = fig.gca() 
# plot mean and 2σ intervals
plt.plot(X_new, mean, 'r', lw=2, label="mean and 2σ region");
plt.plot(X_new, mean + 2*sd, 'r', lw=1); 
plt.plot(X_new, mean - 2*sd, 'r', lw=1);
plt.fill_between(X_new.flatten(), mean - 2*sd, mean + 2*sd, color="r", alpha=0.5) 
# plot original data and true function
plt.plot(X, series, 'ok', ms=3, alpha=1.0, label="observed data"); 
plt.title("predictive mean and σ interval"); plt.legend();

#%% start leave one out routine
from copy import deepcopy
series_original = join_components('add', period, trend, noise) 
X_original = np.array([i for i in range(len(series_original))])[:,None]


from scipy.stats import percentileofscore

score_percentile = list()
for idx in range(len(series)):
    series = deepcopy(series_original)
    X = np.array([i for i in range(len(series)) if i != idx])[:,None]
    series = np.concatenate((series[:idx], series[idx+1:]), axis=0)
     
    with pm.Model() as ts_gp:
        coef = pm.HalfNormal('coef', sigma=0.5) 
        lin_func = pm.gp.mean.Linear(coeffs=coef, intercept=0.0)
        
        ls_matern = pm.HalfNormal('ls_matern', sigma=0.5)
        cov = pm.gp.cov.Matern52(1, ls=ls_matern) # base for additive gp
        ls_period = pm.HalfNormal('ls_period', sigma=0.5)
        cov += pm.gp.cov.Periodic(1, period=12, ls=ls_period)     
        
        sigma = pm.HalfCauchy("sigma", beta=1)      
        
        gp = pm.gp.Marginal(mean_func=lin_func, cov_func=cov) 
        gp.marginal_likelihood("y", X=X, y=series, noise=sigma) 
        
        trace = pm.sample(5_000, tune=2_000, chains=1, progressbar=True)
    
    forecasted_years = 0
    n_new = 12 * forecasted_years
    X_new = np.array(list(range(0, len(series) + n_new)))[:,None]
    with ts_gp:
            f_pred = gp.conditional("f_pred", X_new)
            pred_samples = pm.sample_posterior_predictive(trace, 
                                vars=[f_pred], samples=2_000)

    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()    
    plot_gp_dist(ax, pred_samples["f_pred"], X_new); 
    # plt.plot(X_original, series_original, 'x', ms=5, alpha=1, label="Observations") 
    plt.axvline(idx) 
    plt.plot(X, series, 'x', ms=5, alpha=1, label="Observations") 

    plt.axhline(series_original[idx])
    plt.title(f"Posterior distribution @ noise={noise_magnitude}"); plt.legend();
    plt.show()

    pred = pred_samples['f_pred']  
    quantity = series_original[idx]
    pct = percentileofscore(pred[:,idx], quantity)
    
    score_percentile.append([idx, quantity, pct])