
# %%
import matplotlib.pylab as plt
from sklearn.datasets import load_iris


# %%
iris=load_iris()

# %%
dir(iris)

# %%
iris.data[1]

# %%
iris.target[0:100]

# %%
iris.target_names[0:100]

# %%
iris.feature_names

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(iris.data,iris.target,test_size=0.3)


# %%
from sklearn.linear_model import LogisticRegression

# %%
model=LogisticRegression()
model.fit(X_train,y_train)

# %%
y_test

# %%
y_predicted=model.predict(X_test)

# %%
model.score(X_test,y_predicted)

# %%
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)


# %%

cm


# %%
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Truth')
plt.ylabel('predicted')

# %%
iris.target_names

# %%



