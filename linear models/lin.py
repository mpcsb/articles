#%%
#https://www.quora.com/What-are-the-advantages-of-logistic-regression-over-decision-trees-Are-there-any-cases-where-its-better-to-use-logistic-regression-instead-of-decision-trees/answer/Claudia-Perlich?ch=10&share=ef233af4&srid=28C3J

import matplotlib.pyplot as plt 

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC


for size in [250,1000,2500,5000,7500,10000,25000]:
    results=[]
    for noise in [1,31,51,61,81,91,100]:
        data, target = make_classification(
            n_samples=size,
#            n_features=15,
#            n_informative=2,
#            n_redundant=0,
            scale=1,
            flip_y=noise*0.01,
            random_state=1987
        )
    
        res1 = cross_val_score(LR(solver='saga',#penalty='elasticnet',
                                  max_iter=int(1e6), warm_start=True), 
                              data, target, scoring='roc_auc', cv=4).mean()
        
        res2 = cross_val_score(GBC( max_depth=5,learning_rate=0.1),
                              data, target, scoring='roc_auc', cv=4).mean()

        res3 = cross_val_score(SVC(gamma='auto'),
                              data, target, scoring='roc_auc', cv=4).mean()
        
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
    ax.set_title('Size: '+ str(size))
    ax.set_xticks([], minor=False)
    ax.set_xlabel('SNR (0-100%)')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    ax.set_ylabel('ROC AUC')
    plt.show()
