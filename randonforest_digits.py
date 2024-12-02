# %%
import pandas as pd

# %%
from sklearn.datasets import load_digits

# %%
digits =load_digits()

# %%
dir(digits)

# %%
df=pd.DataFrame(digits.data)

# %%
df['target']=digits.target

# %%
df

# %%
df['target']=digits.target
df.head()

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)

# %%
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(

)
model.fit(X_train,y_train)

# %%
model.score(X_test,y_test)

# %%
y_predicted=model.predict(X_test)

# %%
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_predicted)
cm

# %%
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

# %%
plt.figure(figsize=(10,7))
sb.heatmap(cm,annot=True)
plt.xlabel('truth')
plt.ylabel('predicted')

# %%



