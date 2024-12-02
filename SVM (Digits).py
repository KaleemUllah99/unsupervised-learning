# %%
import pandas as pd

# %%
from sklearn.datasets import load_digits

# %%
digits =load_digits()

# %%
dir(digits)

# %%
digits.data

# %%
digits.target

# %%
digits.target_names

# %%
df=pd.DataFrame(digits.data,columns=digits.feature_names)

# %%
df['target']=digits.target

# %%
df

# %%
df['Digit_names']=df.target.apply(lambda x:digits.target_names[x])

# %%
df

# %%
x=df.drop(['target','Digit_names'],axis='columns')

# %%
y=df['target']

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2)

# %%
from sklearn.svm import SVC

# %%
model=SVC()

# %%
model.fit(X_train,y_train)

# %%
model.score(X_test,y_test)

# %%
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import axis3d


# %%
df0= df[df.target==0]
df1= df[df.target==1]
df2= df[df.target==2]
df3= df[df.target==3]
df4= df[df.target==4]
df5= df[df.target==5]
df6= df[df.target==6]
df7= df[df.target==7]

# %%
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Scatter plots for the first set of pixels in green
ax.scatter(df0['pixel_0_0'], df0['pixel_0_1'], df0['pixel_0_2'], color='green', marker='o')
ax.scatter(df0['pixel_0_3'], df0['pixel_0_4'], df0['pixel_0_5'], color='green', marker='o')
ax.scatter(df0['pixel_0_6'], df0['pixel_0_7'], df0['pixel_0_6'], color='green', marker='o')

ax.scatter(df1['pixel_0_0'], df1['pixel_0_1'], df1['pixel_0_2'], color='red', marker='+')
ax.scatter(df1['pixel_0_3'], df1['pixel_0_4'], df1['pixel_0_5'], color='red', marker='+')
ax.scatter(df1['pixel_0_6'], df1['pixel_0_7'], df1['pixel_0_6'], color='red', marker='+')

############################################################################
# Scatter plots for the second set of pixels in red
ax.scatter(df0['pixel_1_0'], df0['pixel_1_1'], df0['pixel_1_2'], color='green', marker='+')
ax.scatter(df0['pixel_1_3'], df0['pixel_1_4'], df0['pixel_1_5'], color='green', marker='+')
ax.scatter(df0['pixel_1_6'], df0['pixel_1_7'], df0['pixel_1_6'], color='green', marker='+')

# Set labels for axes
ax.set_xlabel('X Axis (Pixels)')
ax.set_ylabel('Y Axis (Pixels)')
ax.set_zlabel('Z Axis (Pixels)')



# %%
df0.size

# %%
df1.size

# %%
df['pixel_0_0'].size

# %%
df['pixel_0_1'].size

# %%



