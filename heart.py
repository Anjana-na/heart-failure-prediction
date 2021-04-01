import pandas as pd
df = pd.read_csv("heart.csv")

df.head()

y = df.DEATH_EVENT
x = df.drop("DEATH_EVENT",axis = "columns")


x.columns[x.isna().any()]

from sklearn import tree


model = tree.DecisionTreeClassifier()

model.fit(x,y)

model.score(x,y)


model.predict([[75.0,0,582,0,20,1,265000.00,1.9,130,1,0,4]])

model.predict([[50.0,0,196,0,45,0,395000.00,1.6,136,1,1,285]])

model.predict([[49,1,80,0,30,1,427000,1,138,0,0,12]])

import pickle

with open("heart_pickle","wb") as f:
    pickle.dump(model,f)



