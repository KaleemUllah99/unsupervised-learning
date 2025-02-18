# %%
from sklearn.datasets import load_digits
digits=load_digits()
dir(digits)

# %%
# create dictionary for models and params defining
#then loop through the dictionary and fit the model and get the score
#score the model and get the best model


# %%

    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier   


# %%
model_params={
 'svm':{
     'model':svm.SVC(gamma='auto'),
     'params':{
            'C':[1,10,20],
            'kernel':['rbf','linear']
     }
    },
 
 'LogisticRegression':{
        'model':LogisticRegression(solver='liblinear'),
        'params':{
            'C':[1,5,10]
        }
 },
 'DecisionTreeClassifier':{
            'model':DecisionTreeClassifier(),
            'params':{
                'criterion':['gini','entropy'],
               
            }
 },
 
 ' RandomForestClassifier':{
            'model':RandomForestClassifier(),
            'params':{
                'n_estimators':[1,5,10]
            }
 },

 'GaussianNB' :{
            'model':GaussianNB(),
            'params':{
              
            }
            },
   
     
 
 'MultinomialNB':{
            'model':MultinomialNB(),
            'params':{
               
 }
 
     
}

}

# %%

from sklearn.model_selection import GridSearchCV
scores=[]
for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(X,y)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })
    df=pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

# %%



