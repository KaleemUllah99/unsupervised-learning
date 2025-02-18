# %%
import pandas as pd


# %%
df=pd.read_csv('diabetes.csv')
df.head()

# %%
df.isnull().any()

# %%
df.describe()

# %%
df.Outcome.value_counts()

# %%
268/500

# %%
X=df.drop('Outcome',axis=1)
y=df['Outcome']


# %%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X_scaled,y,test_size=0.2,stratify=y,random_state=0)

# %%

ytrain.value_counts()

# %%
214/400


# %%
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)

# %%
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,xtrain,ytrain,cv=5)
scores.mean()

# %%
from sklearn.ensemble import BaggingClassifier
bag_model= BaggingClassifier(
    estimator =DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
bag_model.fit(xtrain,ytrain)
bag_model.oob_score_

# %%
scores=cross_val_score(bag_model,xtrain,ytrain,cv=5)
scores.mean()
bag_model.score(xtest,ytest)

# %%



