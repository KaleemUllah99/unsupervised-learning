# %%
from sklearn import svm,datasets
iris = datasets.load_iris()
dir(iris)
import numpy as np

# %%
import pandas as pd
X=iris.data
df= pd.DataFrame(X,columns=iris.feature_names)
df['flower']=iris.target
df['flower']=df['flower'].apply(l ambda x: iris.target_names[x] )
df[0:5]

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(iris.data,iris.target,test_size=0.3)

# %%
model=svm.SVC(kernel='rbf',C=30,gamma='auto')
model.fit(xtrain,ytrain)
model.score(xtest,ytest)

# %%
 from sklearn.model_selection import cross_val_score
 

# %%
cross_val_score(svm.SVC(kernel='linear',C=10,gamma='auto'),iris.data,iris.target,cv=5)

# %%
cross_val_score(svm.SVC(kernel='rbf',C=10,gamma='auto'),iris.data,iris.target,cv=5)

# %%
cross_val_score(svm.SVC(kernel='linear',C=20,gamma='auto'),iris.data,iris.target,cv=5)

# %%
kernels=['rbf','linear']
C=[1,10,20]
avg_scores={}
for kval in kernels:
    for cval in C:
        cv_scores=cross_val_score(svm.SVC(kernel=kval,C=cval,gamma='auto'),iris.data,iris.target,cv=5)
        avg_scores[kval +'_'+str(cval)]=np.average(cv_scores)
avg_scores

# %%
from sklearn.model_selection import GridSearchCV


# %%
clf=GridSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
    
},cv=5,return_train_score=False)
clf.fit(iris.data,iris.target)
clf.cv_results_

# %%
df=pd.DataFrame(clf.cv_results_)
df

# %%
clf.bes
t_score_

# %%
clf.best_params_

# %%
from sklearn.model_selection import RandomizedSearchCV
rs=RandomizedSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']   
    },cv=5,
    return_train_score=False,
    n_iter=2
                      )
rs.fit(iris.data,iris.target)
pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]

# %%



