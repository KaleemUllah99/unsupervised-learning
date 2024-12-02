# %%
import pandas as pd
df=pd.read_csv('insurance_data.csv')

# %%
df

# %%
df.head()

# %%
import matplotlib.pyplot as plt
plt.scatter(df.age,df.bought_insurance,marker='+',color='red')


# %% [markdown]
# df.shape
# 

# %%
from sklearn.model_selection import train_test_split

# %%
x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)

# %%
x_test

# %%
from sklearn.linear_model import LogisticRegression

# %%
model=LogisticRegression()

# %%
model.fit(x_train,y_train)

# %%
model.predict(x_test)

# %%
x_test

# %%
model.score(x_test,y_test)

# %%
model.predict_proba(x_test)

# %%
model.predict ([[40.445]])

# %% [markdown]
# 

# %%



