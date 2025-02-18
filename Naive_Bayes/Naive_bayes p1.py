# %%
import pandas as pd
df =pd.read_csv('titanic.csv')
df.head()

# %%
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis='columns',inplace=True)
df.head()

# %%
df.head()

# %%
target=df.Survived
inputs=df.drop('Survived',axis='columns')

# %%
df.head()

# %%
inputs.head()

# %%
dummies=pd.get_dummies(inputs.Sex).astype('int64')
dummies.head(3)

# %%
inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head()

# %%
inputs=inputs.drop('Sex',axis='columns')
inputs.head()

# %%
inputs.columns[inputs.isna().any()]


# %%
inputs.Age[:10]

# %%
inputs.Age=inputs.Age.fillna(inputs.Age.mean())
inputs.Age[:10]

# %%
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split (inputs,target,test_size=0.2)


# %%
len(xtrain)

# %%
len(xtest)

# %%
len(inputs)

# %%
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

# %%
model.fit(xtrain,ytrain)

# %%
model.score(xtest,ytest)

# %%
xtest[0:10]

# %%
ytest[0:10]

# %%
model.predict(xtest[0:10])

# %%


# %% [markdown]
# 


