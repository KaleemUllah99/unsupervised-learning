# %%
import pandas as pd
df=pd.read_csv('spam.csv')
df.head()

# %%
df.groupby('Category').describe()

# %%
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()

# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(df.Message,df.spam,  test_size=0.25)

# %%
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
Xtrain_count=v.fit_transform(xtrain.values)
Xtrain_count.toarray()[:3]


# %%
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(Xtrain_count,ytrain)

# %%
emails=[
    'Hey mohan, can we get together to watch football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count=v.transform(emails)
model.predict(emails_count)

# %%
X_test_count=v.transform(xtest)
model.score(X_test_count,ytest)

# %%
from sklearn.pipeline import Pipeline
clf =Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

# %%
clf.fit(xtrain,ytrain)

# %%
clf.score(xtest,ytest)

# %%
clf.predict(emails)

# %%



