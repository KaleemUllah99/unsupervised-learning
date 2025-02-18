# %%
import pandas as pd
df=pd.read_csv('heart.csv')
df.head()

# %%
def remove_high_outliners(df,column):
    shisui = df[column].mean() + 3*df[column].std()
    return df[df[column]<=shisui]
df1 = remove_high_outliners(df,'Cholesterol')
df2 = remove_high_outliners(df1,'Oldpeak')
df3 = remove_high_outliners(df2,'RestingBP')

# %%
df3.head()

# %%
df[df.Oldpeak>(df.Oldpeak.mean()+3*df.Oldpeak.std())]

# %%
df.Sex.unique() 

# %%
df.ChestPainType.unique()

# %%
df.RestingECG.unique()

# %%
df.ExerciseAngina.unique()  

# %%
df.ST_Slope.unique()

# %%
d4= df3.copy()
d4.head()

# %%
df4["ExerciseAngina"] = df4["ExerciseAngina"].replace({'N': 0, 'Y': 1})

# %%
df4.head()

# %%
df4.ST_Slope=df4.ST_Slope.replace({'Up':1,'Flat':2,'Down':3})

# %%
df4.head()

# %%
df4.RestingECG= df4.RestingECG.replace(
    {
        'Normal': 1,
        'ST': 2,
        'LVH': 3
    })

# %%
df4.head()

# %%
df5= pd.get_dummies(df4, drop_first=True).astype('int64')
df5.head()

# %%
X =df5.drop('HeartDisease',axis=1)
y=df5.HeartDisease
X.head()

# %%
from sklearn.preprocessing import StandardScaler    
scaler= StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:5]

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
xtrain.shape,xtest.shape

# %%
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
scores =cross_val_score(SVC(),X,y,cv=5)
scores.mean()

# %%
from sklearn.ensemble import BaggingClassifier
bag_model =BaggingClassifier(
    estimator=SVC(),
    n_estimators=10,
    max_samples=0.8,
    oob_score=True
)
bag_model.fit(xtrain,ytrain)
bag_model.oob_score_

# %%
bag_model.predict(xtest)

# %%
from sklearn.tree import DecisionTreeClassifier
score=cross_val_score(DecisionTreeClassifier(),X,y,cv=5)
scores.mean()

# %%
bag_model =BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10,
        max_samples=0.8,
        oob_score=True
        
)
bag_model.fit(xtrain,ytrain)
bag_model.oob_score_

# %%


# %%


# %%


# %%


# %%


# %%



