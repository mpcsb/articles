
#%%
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier as RFC
from tqdm import tqdm

import os
path = r'D:\UserData\Z003NJNS\Documents\rec\par opt\viz'
os.chdir(path)


# Load data set and target values
data, target = make_classification(
    n_samples=10000,
    n_features=45,
    n_informative=12,
    n_redundant=7,
    flip_y = 0.2
)


def gen_random_index(n):
    return np.random.choice(data.shape[0], n, replace=False)


def rfccv(dt, min_samples_leaf, min_samples_split, max_features=None):
    val = cross_val_score(
        RFC(min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=min(max_features, 0.999),
            random_state=2,
        ),
        dt, tr, scoring='f1', cv=4, n_jobs=7
    ).mean()
    return val

def lerp(bounds, f):
    a, b = bounds
    return a + f * (b - a)

#%%
samples = 4
size_grid = [100] * samples
steps = 5

grid_search = dict()
with tqdm(total=steps**3 * samples) as pbar:
    for ind, size in enumerate(size_grid):
        grid_search[ind] = list()


        idx = np.random.randint(len(data), size=size)
        dt = data[idx,:]
        tr = target[idx]


        grid = {'min_samples_leaf': (0.0, 0.5),
                 'min_samples_split': (0.0, 0.9),
                 'max_features': (0.0, 0.999),
                }


        for n1 in range(steps):
            msl = round(lerp(grid['min_samples_leaf'], n1/(steps-1)), 3)
            for n2 in range(steps):
                mss = round(lerp(grid['min_samples_split'], n2/(steps-1)), 3)
                for n3 in range(steps):
                    mf = round(lerp(grid['max_features'], n3/(steps-1)), 3)
                    pbar.update(1)

                    val = rfccv(dt=dt,
                                     min_samples_leaf=msl,
                                     min_samples_split=mss,
                                      max_features=mf
                                     )
                    val = np.nan_to_num(val)
                    if val == 0:
                        continue

                    grid_search[ind].append([val, msl, mss, mf])




#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

for ind in range(samples):
    lst = grid_search[ind]

    w_x_y_z = [(float(l[0]), float(l[1]), float(l[2]), float(l[3])) for l in lst]
    w, x, y, z = list(map(list, zip(*w_x_y_z)))

    w = np.array(w)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # ax._axis3don = False
    for i in range(0, 359, 5):
        ax = Axes3D(plt.figure(figsize=(15, 15)))
        ax.scatter(x, y, z, c=w, s=10000, alpha=0.3, marker='s')
        ax.view_init(25, i)
        # ax._axis3don = False
        plt.title(str(size))
        plt.savefig(str(i) + '.png')
        plt.close()

    def getint(name):
        ''' aux function used to sort integer strings'''
        num = name.split('.')[0]
        return int(num)


    images = [f for f in os.listdir(path) if 'png' in f]
    images_sorted = list(sorted(images, key=getint))

    images_lst = []
    for filename in images_sorted:
        images_lst.append(imageio.imread(filename))
    if len(images_lst) > 0:
        kargs = { 'duration': 0.15}
        imageio.mimsave(f'animation_{ind}.gif', images_lst, **kargs)

    for filename in images_sorted :
        os.remove(filename)