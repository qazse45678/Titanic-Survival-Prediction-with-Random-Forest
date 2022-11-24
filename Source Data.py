import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
print(train_data.shape)
print(test_data.columns)

train_data.head()

missing_values = train_data.isnull().sum().sort_values(ascending = False)
missing_values[missing_values > 0]

train_data.info()

numerical_cols = [column for column in train_data.columns if train_data[column].dtype in ['int64', 'float64']]
pd.DataFrame(train_data[numerical_cols].describe())

cat_cols = [column for column in train_data.columns if train_data[column].dtype == 'object']
pd.DataFrame(train_data[cat_cols].describe())

pd.DataFrame(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = ['Survived'], ascending = False))
pd.DataFrame(train_data[['SibSp', 'Survived']]).groupby(['SibSp']).mean().sort_values(by = ['Survived'], ascending = False)
pd.DataFrame(train_data[['Parch', 'Survived']]).groupby('Parch').mean().sort_values(by = ['Survived'], ascending = False)

sns.FacetGrid(train_data, col = 'Survived').map(plt.hist, 'Age', bins = 20)

pd.DataFrame(train_data[['Pclass', 'Survived']]).groupby(['Pclass']).mean().sort_values(by = ['Survived'], ascending = False)

grid = sns.FacetGrid(train_data, col='Survived', row='Pclass')
grid.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_data, row='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived')
grid.map(sns.barplot, 'Sex', 'Fare', ci = None)

missing_values = train_data.isnull().sum().sort_values(ascending = False)
missing_values[missing_values > 0]

train_data = train_data.drop(['Cabin'], axis = 1)
train_data = train_data.drop(['Name', 'Ticket'], axis = 1)

numerical_transformer = SimpleImputer(strategy = 'mean')
Age = train_data['Age'].values.reshape(-1,1)

imputed_age = numerical_transformer.fit_transform(Age)
train_data['Age'] = imputed_age

cat_transformer = SimpleImputer(strategy = 'most_frequent')
embarked = train_data['Embarked'].values.reshape(-1,1)

imputed_embarked = cat_transformer.fit_transform(embarked)
train_data['Embarked'] = imputed_embarked

test_data = test_data.drop(['Cabin'], axis = 1)
test_data = test_data.drop(['Name', 'Ticket'], axis = 1)

numerical_transformer = SimpleImputer(strategy = 'mean')
Age = test_data['Age'].values.reshape(-1,1)
imputed_age_test = numerical_transformer.fit_transform(Age)
test_data['Age'] = imputed_age_test

cat_transformer = SimpleImputer(strategy = 'most_frequent')
embarked = test_data['Embarked'].values.reshape(-1,1)
imputed_embarked_test = cat_transformer.fit_transform(embarked)
test_data['Embarked'] = imputed_embarked_test

numerical_transformer = SimpleImputer(strategy = 'mean')
Fare = test_data['Fare'].values.reshape(-1,1)
imputed_fare_test = numerical_transformer.fit_transform(Fare)
test_data['Fare'] = imputed_fare_test

train_data = train_data.drop('PassengerId', axis = 1)
merge = [train_data, test_data]

for dataset in merge:
    dataset['Sex'] = dataset['Sex'].map({
        'female': 0,
        'male': 1
    })
    
train_data.head()

print(train_data['Embarked'].unique())
print(test_data['Embarked'].unique())

for dataset in merge:
    dataset['Embarked'] = dataset['Embarked'].map({
        'S': 0,
        'C': 1,
        'Q': 2
    })

train_data.head()

X_train = train_data.drop('Survived', axis = 1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis = 1).copy()
#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
model = RandomForestClassifier(n_estimators = 100)

model.fit(X_train, y_train)
predict = model.predict(X_test)
score = round(model.score(X_train, y_train)*100, 2)

print(score)
