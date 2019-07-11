# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:22:17 2018

@author: Miguel
"""

#%%
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
 
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from time import time

from bayes_opt import BayesianOptimization


# Load data set and target values
data, target = make_classification(
    n_samples=100000,
    n_features=45,
    n_informative=12,
    n_redundant=7,
    flip_y = 0.2
)

def gen_random_index(n):
    return np.random.choice(data.shape[0], n, replace=False)  
#%%
sample_size = [500+3000*n for n in range(0,15)]
index_list=[gen_random_index(n) for n in sample_size]


#def svccv(C, gamma):
#    val = cross_val_score(
#        SVC(kernel='rbf', cache_size=4000, C=C, gamma=gamma, random_state=2),
#        dt, tr, 'f1', cv=2
#    ).mean()
#    return val


def rfccv(min_samples_leaf,min_samples_split, max_features):
    val = cross_val_score(
        RFC(min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=min(max_features, 0.999),
            random_state=2,
            
        ),
        dt, tr, scoring='f1', cv=2, n_jobs =7
    ).mean()
    return val


gp_params = {"alpha": 1e-5}


#if __name__ == "__main__":
sv_maximum=[]
t=time()
for size in index_list:
    t0=time()
    
    dt=data[size]
    tr=target[size]
 
    rfcBO = BayesianOptimization(
        rfccv,
        {'min_samples_leaf': (0.1, 0.5),
        'min_samples_split': (0.1, 0.999),
        'max_features': (0.1, 0.999) 
        },verbose=1
    )
    rfcBO.maximize(init_points=50, n_iter=10, **gp_params)
    
#    svcBO = BayesianOptimization(svccv,
#        {'C': (0.001, 100), 'gamma': (0.0001, 0.1)},verbose=1)
 
#    svcBO.maximize(init_points=100, acq="poi", xi=1e-4, n_iter=2, **gp_params)
    print('-' * 53)
 
    print('Final Results','sample size:',len(size))
    print(time()-t0)
    print(len(size),rfcBO.res['max'])
    sv_maximum.append([len(size),rfcBO.res['max'],rfcBO.res['all'],rfcBO.space.X,rfcBO.space.Y])

#    sv_maximum.append([len(size),svcBO.res['max'],svcBO.res['all'],svcBO.space.X,svcBO.space.Y])
print(divmod((time()-t),60))


#%%
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os

#os.chdir('C:\\Users\\Miguel\\Documents\\parameter_estimation')
for i in sv_maximum:
    x=i[3][:,0] 
    np.insert(x, 0, 0)
    y=i[3][:,1] 
    np.insert(y, 0, 0)
    w=i[3][:,2] 
    np.insert(w, 0, 0)
    z=i[4] 
    np.insert(z, 0, 0)
    print(i[0])
#    plt.scatter(x,y,c=z, cmap='plasma') 
#    plt.show()
#        
#    plt.scatter(x,w,c=z, cmap='plasma') 
#    plt.show()
#    
    plt.scatter(y,w,c=z, cmap='plasma') 
#    plt.show()   
    
#    ax = plt.axes(projection='3d')
#    ax.scatter3D(w,x,y, c=z, cmap='plasma', linewidth=10,alpha=.4)
#    plt.savefig(str(i[0])+'fig2.png',dpi=150)
 