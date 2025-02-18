# %%
 import pandas as pd
 from sklearn import datasets
 iris=datasets.load_iris()
 digits=datasets.load_digits()
 
 

# %%
X,y = iris.data, iris.target
df =pd.DataFrame(X,columns=iris.feature_names)
df['target']=iris.target
df['species']=df.target.apply(lambda x: iris.target_names[x])
df.head()

# %%
len(df)

# %%
import matplotlib.pyplot as plt
df_0= df[df.target==0]
plt.scatter(df_0['sepal length (cm)'],df_0['sepal width (cm)'],color='red',marker='+')
df_1 =df[df.target==1]
plt.scatter(df_1['sepal length (cm)'],df_1['sepal width (cm)'],color='blue',marker='.')
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.show()

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(iris.data,iris.target,test_size=0.2,random_state=1)


# %%
len(xtrain)

# %%
len(ytrain)

# %%
from sklearn.neighbors import KNeighborsClassifier
Knn=KNeighborsClassifier(n_neighbors=3)
Knn.fit(xtrain,ytrain)

# %%
Knn.score(xtest,ytest)

# %%
digits=load_digits()
A,b= digits.data,digits.target
df_d=pd.DataFrame(A,columns=digits.feature_names)
df_d['target']=b
df_d.head()

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(A,b,test_size=0.2,random_state=1)


# %%
len(xtrain)

# %%
len(ytrain)

# %%
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=10)
knn.fit(xtrain,ytrain)

# %%
knn.score(xtest,ytest)

# %%
model_params={
    'KN':{
        'model':  KNeighborsClassifier(),
        'param':{
            'n_neighbors':[3,5,6,8,10]
        }
    }
}

# %%
from sklearn.model_selection import GridSearchCV
scores={}
for names,config in model_params.items():
    clf=GridSearchCV(config['model'],config['param'],cv=5,return_train_score =False)
    clf.fit(xtrain,ytrain)
    scores[names]={
        'best_params':clf.best_params_,
        'best_scores':clf.best_score_
    }
print(scores)


# %%
from sklearn.metrics import classification_report
y_pred=knn.predict(xtest)
print(classification_report(ytest,y_pred))


