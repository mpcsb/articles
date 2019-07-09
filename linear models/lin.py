from random import randint

import matplotlib.pyplot as plt 

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR

make_classification()
data, target = make_classification(
    n_samples=300,
    n_features=23,
    n_informative=20,
#    n_redundant=2,
    scale=1,
    flip_y=0.1
)

 # target with noise
import copy
results=[]
#print(sum(target))

for noise in range(1,100,5):
    noisy_target=copy.deepcopy(target)
    
    for i in range(len(target)):
        if randint(0,100)>noise: 
            if noisy_target[i]==0:
                noisy_target[i]=1

    res1= cross_val_score(LR( solver='saga',max_iter=int(1e6), warm_start=True), 
                          data, noisy_target, scoring='f1', cv=4).mean()
    #gradient boost
    res2= cross_val_score(GBC( max_depth=3,learning_rate=0.05),
                          data, noisy_target, scoring='f1', cv=4).mean()
    
    results.append([res1,res2])
    print(results[-1])

a,b=[],[]
for i in results:
    a.append(i[0])
    b.append(i[1]) 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(a,'r--',label='Lin Reg')
ax.plot(b,'b--',label='GBM') 
#ax.set_xlabel('SNR (0-100%)')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
ax.set_ylabel('F1 score')
plt.show()

#%%
import copy

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
 
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
#from random import randint

features=25
data, target = make_classification(
    n_samples=1000,
    n_features=features,
    n_informative=23,
    n_redundant=2
)

#def remove_collinearity(input_data, threshold):
#    delete_cols=set()
#    
#    for i in range(input_data.shape[1]):
#        for j in range(input_data.shape[1]-1):
#            if i > j:
#                if  np.corrcoef(input_data[:,i],input_data[:,j])[0][1]>threshold:
#                    delete_cols.add(i)
##                    print(i,j, np.corrcoef(data[:,i],data[:,j])[0][1])
#    
#    delete_cols = list(delete_cols)
#    
#    
#    datax=copy.deepcopy(input_data)
#    for i in reversed(delete_cols):
#        datax = np.delete(datax, i, 1)
#        datax=copy.deepcopy(datax)
#    
#    return datax#, delete_cols

#def new_features(input_data):
#    temp1=np.arange(len(input_data))
#    temp2=np.arange(len(input_data))
#    temp3=np.arange(len(input_data))
#    temp4=np.arange(len(input_data))
#    temp5=np.arange(len(input_data))
#    ss=copy.deepcopy(input_data)
#    for i in range(features):
#        for j in range(features):
#            if i>j:
#                temp1 = data[:,i]-data[:,j]
#                temp2 = data[:,i]+data[:,j]
#                temp3 = data[:,i]*data[:,j]
#                temp4 = data[:,i]/data[:,j]
#                temp5 = np.sqrt(np.abs(temp3))
#                
#        ss=np.column_stack((ss,temp1))
#        ss=np.column_stack((ss,temp2))        
#        ss=np.column_stack((ss,temp3))
#        ss=np.column_stack((ss,temp4))   
#        ss=np.column_stack((ss,temp5))   
#    
#    return ss



results=[]


for n in range(1,100,10):
    noisy_data = copy.deepcopy(data)
    
    for i in range(data.shape[1]):
        noise = np.random.rand(1, data.shape[1])*n
        noisy_data[i] = data[i]*noise
    
    #gradient boost
    res2= cross_val_score(GBC(max_depth=6),
                          noisy_data, target, cv=3).mean()       
    
#    noisy_data = new_features(noisy_data)   
#    noisy_data = new_features(noisy_data)   

    #LR
    res1= cross_val_score(LR(penalty='l2',solver='liblinear',tol=1e-4, 
                          max_iter=int(1e6), warm_start=True),
                          noisy_data, target, cv=3).mean()
        
    results.append([res1,res2])
    print(n,results[-1])
    

a,b=[],[]

for index, i in enumerate(results):
    a.append(i[0])
    b.append(i[1]) 
    
fig = plt.figure( figsize=(13, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(a,'r-',b,'b-')
ax.set_xlabel('SNR')
ax.set_ylabel('F1 score')
plt.show()
#%%

from random import randint

import matplotlib.pyplot as plt 

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR


data, target = make_classification(
    n_samples=5000,
    n_features=50,
    n_informative=20,
    scale=1,
    flip_y=0.1
)

 # target with noise
import copy
results=[]


for noise in range(1,100,10):
    
    data, target = make_classification(
        n_samples=1000,
        n_features=53,
        n_informative=20,
        n_redundant=10,
        scale=1,
        flip_y=noise*0.01,
        random_state=1
    )

    res1= cross_val_score(LR( solver='saga',max_iter=int(1e6), warm_start=True), 
                          data, target, scoring='f1', cv=4).mean()
    
    res2= cross_val_score(GBC( max_depth=5,learning_rate=0.05),
                          data, target, scoring='f1', cv=4).mean()
    
    results.append([res1,res2])
    print(results[-1],noise*0.01)

a,b=[],[]
for i in results:
    a.append(i[0])
    b.append(i[1]) 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(a,'r-',label='Lin Reg')
ax.plot(b,'b-',label='GBM') 
#ax.set_xlabel('SNR (0-100%)')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
ax.set_ylabel('F1 score')
plt.show()
