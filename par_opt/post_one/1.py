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
# from time import time

# Load data set and target values
data, target = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=12,
    n_redundant=7,
    # flip_y = 0.2
)

def gen_random_index(n):
    return np.random.choice(data.shape[0], n, replace=False)  
 

def rfccv(dt, min_samples_leaf, min_samples_split, max_features=None):
    val = cross_val_score(
        RFC(min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            # max_features=min(max_features, 0.999),
            random_state=2,
        ),
        dt, tr, scoring='accuracy', cv=4, n_jobs=7
    ).mean()
    return val

def lerp(a, b, f):
    return a + f * (b - a)
 
grid_search = dict()
size_grid = [250, 500, 1000, 2000, 5000, 7500]
for size in size_grid:
    print(size)
    grid_search[size] = list() 
    
    idx = np.random.randint(len(data), size=size)
    dt = data[idx,:]
    tr = target[idx]
    
    
    # grid = {'min_samples_leaf': (0.0, 0.5),
    #          'min_samples_split': (0.0, 0.9),
    #          'max_features': (0.0, 0.999), 
    #         }
    grid = {'min_samples_leaf': (0.01, 0.35),
         'min_samples_split': (0.01, 0.65), 
        }    
    steps = 10
    for n1 in range(steps):
        msl = round(lerp(grid['min_samples_leaf'][0], grid['min_samples_leaf'][1], n1/(steps-1)), 3)
        for n2 in range(steps):
            mss = round(lerp(grid['min_samples_split'][0], grid['min_samples_split'][1], n2/(steps-1)), 3)
            # for n3 in range(steps):
            #     mf = round(lerp(grid['max_features'][0], grid['max_features'][1], n3/(steps-1)), 3)
         
            val =  rfccv(dt=dt, 
                             min_samples_leaf=msl, 
                             min_samples_split=mss, 
                              # max_features=mf
                             )   
            grid_search[size].append([val, msl, mss])
    print(max(grid_search[size]))
    

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib as mpl


colormap = mpl.cm.autumn 
colorst = [colormap(55*i) for i in range(len(size_grid))]
title_font = {'fontname':'Arial', 'size':'26', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}


for i, size in enumerate(size_grid):
    lst = grid_search[size] 
    
    w_x_y = [(float(l[0]), float(l[1]), float(l[2]) ) for l in lst] 
    w, x, y  = list(map(list, zip(*w_x_y ))) 

    w = np.array(w)
    x = np.array(x) 
    y = np.array(y) 
    # z = np.array(z) 
         
    # ax._axis3don = False
    for degree in range(60, 420, 360): 
        ax = Axes3D(plt.figure(figsize=(15, 15)))
        # ax.scatter(x, y, w, s=1000, marker='o')
        ax.plot_trisurf(y, x, w, alpha=0.6, color=colorst[i])

        ax.view_init(25, degree)   
        ax.set_ylabel('min_samples_leaf')
        ax.set_xlabel('min_samples_split')
        plt.title('Sample size: ' + str(round(size/100))+ '%', **title_font) 

        plt.show()
#%%
    
import os 
import matplotlib.pyplot as plt

import numpy as np
import csv
import imageio

path = r'/home/miguel/Documents/projects/Wildfire/wfire/src/simulation/viz'
os.chdir(path)

try:
    os.remove('animation.gif')
except:
    pass

def getint(name): 
    ''' aux function used to sort integer strings'''
    num = name.split('.')[0] 
    return int(num)


def plot3d(f, i): 
    colors = {'road':'black',
              'water':'blue',
              'tree':'green',
              'burning_tree':'red',
              'ember':'orange',
              'ash':'grey'}

    content = list()
    with open(f) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            content.append(row[0].split(',')) 

    x_y_z = [(float(l[0]), float(l[1]), float(l[2])) for l in content[1:]]
    color = [colors[l[3]] for l in content[1:]]  
    x, y, z = list(map(list, zip(*x_y_z))) 


    x = np.array(x)#.reshape(self.terrain.num_points, self.terrain.num_points)
    y = np.array(y)#.reshape(self.terrain.num_points, self.terrain.num_points)
    z = np.array(z)#.reshape(self.terrain.num_points, self.terrain.num_points)
        
        
    ax = Axes3D(plt.figure(figsize=(15, 15)))
    ax.scatter(x, y, z, c=color, marker='^')
    ax._axis3don = False
    ax.view_init(35, 60 + i*10)  
    plt.savefig(str(i) + '.png')
    plt.close()



files = [f for f in os.listdir(path)]
files_sorted = list(sorted(files, key=getint)) 

for i, f in enumerate(files_sorted): 
    plot3d(f, i) 

images = [f for f in os.listdir(path) if 'png' in f]
images_sorted = list(sorted(images, key=getint))

images_lst = []
for filename in images_sorted:
    images_lst.append(imageio.imread(filename))
if len(images_lst) > 0:
    kargs = { 'duration': 0.15}
    imageio.mimsave('animation.gif', images_lst, **kargs) 

for filename in images_sorted + files_sorted:
    os.remove(filename)