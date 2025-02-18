# %%
from sklearn.datasets import load_wine
import pandas as pd
wine=load_wine()
X,y=wine.data,wine.target
df=pd.DataFrame(X,columns=wine.feature_names)
df.head()

# %%
df['target']=y
df.head()

# %%
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X_scaled[:5]

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X_scaled,y,test_size=0.25) 


# %%
xtrain.shape,xtest.shape

# %%
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)

# %%
model.score(xtest,ytest)

# %%
model.predict(xtest)

# %%
wine.


