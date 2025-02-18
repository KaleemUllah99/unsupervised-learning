# %%
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()
digits.data[0]
digits.data[0].reshape(8, 8)

# %%
import matplotlib.pyplot as plt 
plt.gray()
plt.matshow(digits.images[2])   

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(digits.data)

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_scaled,digits.target,test_size=0.2,random_state=0)


# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()    
model.fit(xtrain,ytrain)
model.score(xtest,ytest)

# %%
from sklearn.decomposition import PCA
pca=PCA(0.97)
X_pca=pca.fit_transform(digits.data)


# %%
xtrain,xtest,ytrain,ytest = train_test_split(X_pca,digits.target,test_size=0.2,random_state=0)
model = LogisticRegression()
model.fit(xtrain,ytrain)
model.score(xtest,ytest)

# %%
pca.explained_variance_ratio_


