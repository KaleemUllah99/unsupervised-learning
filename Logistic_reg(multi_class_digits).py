# %%
import matplotlib.pylab as plt
from sklearn.datasets import load_digits


# %%
digits=load_digits()

# %%
dir(digits)

# %%
digits.data[0]

# %%
for i in  range(5):
 plt.matshow(digits.images[i])


# %%
digits.target[0:5]

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.3)

# %%
len(X_train)

# %%
len(X_test)

# %%
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)



# %%
model.score(X_test,y_test)

# %%
plt.matshow(digits.images[67])

# %%
digits.target[67]

# %%
model.predict(digits.data[0:9])

# %%
y_predicted= model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,  y_predicted)
cm

# %%
import seaborn as sn
plt.Figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("truth")


# %%



