


# %% 0 - import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# %% 0 - import data and check the head
data = pd.read_csv("../../data/titanic/train.csv")
data.head()


# %% 0 - convert columns to categories / factors
data['Pclass']   = data['Pclass'].astype('category')
data['Sex']      = data['Sex'].astype('category')
data['Embarked'] = data['Embarked'].astype('category')
# data.dtypes



# %% 0 - remove punctuation, etc. from name
data['Name'] = data['Name'].apply(lambda n: " ".join([v.strip(",()[].\"'") for v in n.split(" ")]))


# %% 0 - 
data['Ticket_No'] = data['Ticket'.apply(lambda t: t.split(" ")[-1])]



# %% 0 - 
data.info()


# %% 0 - 
# data.groupby('Survived')['Pclass'].agg(['count', 'median'])


# %% 0 - 
data.groupby('Survived')['Age'].agg(['count', 'mean', 'median', 'std'])


# %% 0 - 
data.groupby('Survived')['Fare'].agg(['count', 'mean', 'median', 'std'])




# %% 4 - 
X = pd.get_dummies(data[["Pclass", "Sex", "SibSp", "Parch"]])
y = data["Survived"]




# %% 5 - 
# RepeatedKFold(n_splits = 5, n_repeats = 5)
pred = cross_val_predict(LogisticRegression(), X, y, cv = 5)
print(accuracy_score(y, pred))

# %% 5 - 
print(classification_report(y, pred))

