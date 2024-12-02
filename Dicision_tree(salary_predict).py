# %%
import pandas as pd
df=pd.read_csv('salaries.csv')
df
            

# %%
input = df.drop('salary_more_then_100k',axis='columns')

# %%
target=df['salary_more_then_100k']

# %%
input.head()

# %%
from sklearn.preprocessing import LabelEncoder


# %%
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

# %%
input['company_n']      =le_company.fit_transform(input['company'])
input['job_n']          =le_company.fit_transform(input['job'])
input['degree_n']       =le_company.fit_transform(input['degree'])
input.head()

# %%
inputs_n = input.drop(['company','job','degree'], axis='columns')

# %%
inputs_n

# %%
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

# %%
model.fit(inputs_n,target)

# %%
model.score(inputs_n,target)

# %%
model.predict([[2,2,1]])

# %%



