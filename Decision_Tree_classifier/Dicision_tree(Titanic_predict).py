# %%
import pandas as pd

# %%
df=pd.read_csv('titanic.csv')
df

# %%
inputs=df.drop(['Survived','PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')

# %%
inputs.head()

# %%
inputs

# %%
y=0
inputs['Age']=inputs.groupby('Pclass')['Age'].transform(lambda x:x.fillna(x.median()))

# %%
inputs

# %%
df.head()

# %%
target=df['Survived']

# %%
target.head()

# %%
inputs.head()

# %%
from sklearn.preprocessing import LabelEncoder
le_Sex=LabelEncoder()

# %%
inputs['Sex_n']=le_Sex.fit_transform(inputs['Sex'])

# %%
inputs.head()

# %%
inputs=inputs.drop('Sex',axis='columns')

# %%
inputs

# %%
target

# %%


# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.29)

# %%
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# %%
model.fit(x_train,y_train)

# %%
model.score(x_test,y_test)

# %%
model.predict([[3	,35.0,	8.0500,	1]])

# %%


# %%



