#%%
#https://www.quora.com/What-are-the-advantages-of-logistic-regression-over-decision-trees-Are-there-any-cases-where-its-better-to-use-logistic-regression-instead-of-decision-trees/answer/Claudia-Perlich?ch=10&share=ef233af4&srid=28C3J

import matplotlib.pyplot as plt 

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC


for size in [100,250,500,5000,20000]:
    results=[]
    for noise in range(1,101,10):
        data, target = make_classification(
            n_samples=size,
#            n_features=53,
#            n_informative=20,
#            n_redundant=10,
            scale=1,
            flip_y=noise*0.01,
            random_state=1
        )
    
        res1 = cross_val_score(LR( solver='saga',max_iter=int(1e6), warm_start=True), 
                              data, target, scoring='f1', cv=4).mean()
        
        res2 = cross_val_score(GBC( max_depth=5,learning_rate=0.05),
                              data, target, scoring='f1', cv=4).mean()

        res3 = cross_val_score(SVC(gamma='scale'),
                              data, target, scoring='f1', cv=4).mean()
        
        results.append([res1,res2,res3])
#        print(results[-1],noise*0.01)
    
    a,b,c=[],[],[]
    for i in results:
        a.append(i[0])
        b.append(i[1])
        c.append(i[2])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(a,'r-',label='Lin Reg')
    ax.plot(b,'b-',label='GBM') 
    ax.plot(c,'g-',label='SVC') 
    ax.set_title('Sample size: '+ str(size))
    ax.set_xticks([], minor=False)
    ax.set_xlabel('SNR (0-100%)')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    ax.set_ylabel('F1 score')
    plt.show()
