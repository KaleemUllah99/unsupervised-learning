# %%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing  import MinMaxScaler
from sklearn.cluster import KMeans 
#%matplotlib inline


# %%
df=pd.read_csv("income.csv")
df.head()

# %%
plt.scatter(df.Age,df['Income($)'])

# %%
kn = KMeans(n_clusters=3)

# %%
y_prediction = kn.fit_predict(df[['Age','Income($)']])
y_prediction


# %%
df['cluster']=y_prediction

df.head()

# %%

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')

#plt.scatter (kn.cluster_centers_[:,0],kn.clusters_[:,1],color='purple',marker='x',label='ceentroid' )

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

# %%
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])
df


# %%
scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
df.head()


# %%
km=KMeans(n_clusters=3)
y_prediction = km.fit_predict(df[['Age','Income($)']])
y_prediction

# %%
df['cluster'] =y_prediction
#df.drop('cluster', axis='columns',inplace=True)
df

# %%
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')

plt.scatter (kn.cluster_centers_[:,0],kn.cluster_centers_[:,1],color='purple',marker='x',label='ceentroid' )

plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

# %%
kn.cluster_centers_


# %%
k_rng= range(1,10)
sse=[]
for k in k_rng:
    kn= KMeans(n_clusters=k)
    kn.fit(df[['Age','Income($)']])
    sse.append(kn.inertia_)
    

# %%
sse

# %%
plt.xlabel('k')
plt.ylabel('Sum of squared errors')
plt.plot(k_rng,sse)


# %%



