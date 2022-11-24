# Goal of the analysis

1. Define the question
2. Preprocessing
    * Analyze by describing data
    * clean missing values
    * clean outliers
    * transform categorical value
3. Apply model  
    * train-test split
    * find optimal tree leaves with lowest MAE
4. Predict the test set

Time Spent: 2 weeks (2022.11)

# 1. Define the question
**The Titanic dataset** is one of the most popular datasets on Kaggle. We're going to use it to explain the relations between passengers' attributes and their survival. For example, does the family size (numbers of siblings, parents, spouses and children) affects survival? Or how is the class of passengers related to survival?
Also, we'll develop a **random forest** model to predict the survival for given dataset.
# 2. Preprocessing
Before starting, let's import the necessary files from the libraries.

```
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
```
## Analyze by describing data
We'll do basic EDA (exploratory data analysis) by (1)describing the info of the dataset and (2)asking questions to each feature. We'll find the pattern of each feature by showing the statistical metrics, grouping and sorting values to compare with their survival and drawing graphes.

**What features do the data have? How many rows in the data?**

```
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

print(train_data.shape)
print(test_data.columns)
```
<img width="563" alt="image" src="https://user-images.githubusercontent.com/63503783/203753517-9a9d3538-b092-4eab-956a-fd4cb9ea8be2.png">

**Which features are categorical?**

These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.

- Categorical: Survived, Sex, and Embarked
- Ordinal: Pclass

**Which features are numerical?**

Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.

- Continous: Age, Fare
- Discrete: SibSp, Parch
- Ordinal: Pclass

understand the definition of each column
```
train_data.head()
```

**What features have missing values?**
Cabin > Age > Embarked

```
missing_values = train_data.isnull().sum().sort_values(ascending = False)
missing_values[missing_values > 0]
```

**What are the data type of the features?**

- 6 features are integer or float
- 5 features are strings (object)

```
train_data.info()
```
<img width="336" alt="image" src="https://user-images.githubusercontent.com/63503783/203754193-3d4e4a92-0b17-4ac1-95cf-7da2e73345cf.png">

**What pattern can we get from the numerical features?**

Continous: Age, Fare
Discrete: SibSp, Parch

1. What's the survival rate? (survivor number/total onboard number)
2. What's the percentage of the passengers that travel with parents or children?
3. What's the percentage of the passengers that travel with siblings or spouses?
4. What's the distribution of fares?
5. What's the distribution of ages?

```
numerical_cols = [column for column in train_data.columns if train_data[column].dtype in ['int64', 'float64']]
pd.DataFrame(train_data[numerical_cols].describe())
```
<img width="632" alt="image" src="https://user-images.githubusercontent.com/63503783/203754378-329ce423-995c-45a7-aefc-f5f3af2ce801.png">

1. What's the survival rate? (survivor number/total onboard number)
> around 38% (mean of **Survived**: 0.384)

2. What's the percentage of the passengers that travel with parents or children?
> most passengers did not have parents or children onboard. (**Parch** is 0 from 25th to 75th, meaning that > 75% of the passengers have 0 Parch)

3. What's the percentage of the passengers that travel with siblings or spouses?
> nearly 25% of the passengers have siblings or spouses onboard (**SibSp** isn't 1 until 75th)

4. What's the distribution of fares?
> few passengers paid significantly high ($512) 

5. What's the distribution of ages?
> passengers are mostly at the ages of 20-38, and few of them is around 80

**What pattern can we get from the categorical features?**

```
cat_cols = [column for column in train_data.columns if train_data[column].dtype == 'object']
pd.DataFrame(train_data[cat_cols].describe())
```
<img width="442" alt="image" src="https://user-images.githubusercontent.com/63503783/203754497-ddaa6dff-aede-4107-9a85-83426a500289.png">

1. Names are unique. No duplicated names.
2. There are two sexes (genders) onboard: male and female. Males take around 65% of all passengers.
3. Ticket has high duplicated (22%) ratio.
4. Cabin has high duplicated ratio, meaning that many passengers may share the same cabins.
5. There are three types of Embarked. S is used by most passengers.

**Analyze the correlation between features and survival**

- Sex
- SibSp
- Parch
- Age
- Pclass
- Embarked

```
pd.DataFrame(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = ['Survived'], ascending = False))
```
<img width="144" alt="image" src="https://user-images.githubusercontent.com/63503783/203754613-0c1b4508-e287-4ba7-971a-514e2d3c1738.png">

Passengers who have more than 3 siblings or spouses have less chance of survival. Among those who have less than 3 siblings or spouses, the surviving rate is highest with 1 sibling or spouse and lowest when they don't have any.

```
pd.DataFrame(train_data[['SibSp', 'Survived']]).groupby(['SibSp']).mean().sort_values(by = ['Survived'], ascending = False)
```
<img width="124" alt="image" src="https://user-images.githubusercontent.com/63503783/203755112-23292579-ef4d-4f26-9641-e561ac1e599e.png">

Passengers who have more than 4 parents or children are rarely survived. Those who have less than 4 parents or children: they have lowest surviving rate with 0 parents and children and over 50% surviving rate with 1 to 3 parents or children.

```
pd.DataFrame(train_data[['Parch', 'Survived']]).groupby('Parch').mean().sort_values(by = ['Survived'], ascending = False)
```
<img width="122" alt="image" src="https://user-images.githubusercontent.com/63503783/203755220-7b4abe6b-942c-4706-a4c5-9d0ea7f56a6f.png">

- Infants (Age <=4) had high survival rate.
- Oldest passengers (Age = 80) survived.
- Large number of 15-25 year olds did not survive.
- Most passengers are in 15-35 age range.

```
sns.FacetGrid(train_data, col = 'Survived').map(plt.hist, 'Age', bins = 20)
```
<img width="438" alt="image" src="https://user-images.githubusercontent.com/63503783/203755414-77ce34f6-a25c-48d2-bfc1-f5c26ef4e425.png">

In general, Pclass 1 has the highest surviving rate, while Pclass 3 has the lowest one.

```
pd.DataFrame(train_data[['Pclass', 'Survived']]).groupby(['Pclass']).mean().sort_values(by = ['Survived'], ascending = False)
```
<img width="128" alt="image" src="https://user-images.githubusercontent.com/63503783/203755504-f3ecb5d2-8abd-434d-9c67-acd67112fc38.png">

Let's compare the survival with **Age** in each Pclass. 

- In Pclass 1, where the surviving rate is the highest among all, passengers who didn't survived don't show particular trend of age. In the same class, passengers who survived were mostly 30~40 years old.

- In Pclass 2 and 3, most of the passengers who didn't survive were between 20~40 years old. It may be because most of the passengers lie in the range of age.

- Pclass 3 has the most passengers, however, most of them didn't survive.

```
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass')
grid.map(plt.hist, 'Age', bins=20)
```
<img width="434" alt="image" src="https://user-images.githubusercontent.com/63503783/203755637-cfc4c4bf-dfad-4919-81c5-0172e1de0e62.png">

- Female have higher surviving rate than male, except for Embarked C.

```
grid = sns.FacetGrid(train_data, row='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
```
<img width="377" alt="image" src="https://user-images.githubusercontent.com/63503783/203755747-75efca1b-56a9-4b5d-8079-e68c1307b3ed.png">

Passengers who have higher **Fare** has higher surviving rate.
```
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived')
grid.map(sns.barplot, 'Sex', 'Fare', ci = None)
```
<img width="441" alt="image" src="https://user-images.githubusercontent.com/63503783/203755871-8bb2195f-814c-4656-ae95-374e62784350.png">

## Clean Missing Values
Find which feature has missing value and the number of missing values.

```
missing_values = train_data.isnull().sum().sort_values(ascending = False)
missing_values[missing_values > 0]
```
<img width="145" alt="image" src="https://user-images.githubusercontent.com/63503783/203755972-95df5695-8167-4e4f-a568-11584051d206.png">

Drop **Cabin** because there are too many missing values within

```
train_data = train_data.drop(['Cabin'], axis = 1)
```

Drop **Name, Ticket** because they don't affect the result.

```
train_data = train_data.drop(['Name', 'Ticket'], axis = 1)
```

Impute mean into the missing values of **Age**.

```
numerical_transformer = SimpleImputer(strategy = 'mean')
Age = train_data['Age'].values.reshape(-1,1)

imputed_age = numerical_transformer.fit_transform(Age)
train_data['Age'] = imputed_age
```

Impute most frequent value into the missing values of **Embarked**.

```
cat_transformer = SimpleImputer(strategy = 'most_frequent')
embarked = train_data['Embarked'].values.reshape(-1,1)

imputed_embarked = cat_transformer.fit_transform(embarked)
train_data['Embarked'] = imputed_embarked
```

Repeat the above steps to clean **Test** dataset.

```
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
```

Impute missing value for **Fare** in Test dataset.

```
numerical_transformer = SimpleImputer(strategy = 'mean')
Fare = test_data['Fare'].values.reshape(-1,1)
imputed_fare_test = numerical_transformer.fit_transform(Fare)
test_data['Fare'] = imputed_fare_test
```

Merge train and test data sets together.

```
train_data = train_data.drop('PassengerId', axis = 1)
merge = [train_data, test_data]
```

## transform categorical value
Map the two types of value, female and male, in **"Sex"** column as 0 and 1.

```
for dataset in merge:
    dataset['Sex'] = dataset['Sex'].map({
        'female': 0,
        'male': 1
    })
    
train_data.head()
```
<img width="448" alt="image" src="https://user-images.githubusercontent.com/63503783/203756634-41d9366e-828c-4c8c-afa8-15aaa9b79df5.png">

Find the total types of values in **'Embarked'** and transform them into numerical data.

```
print(train_data['Embarked'].unique())
print(test_data['Embarked'].unique())
```
<img width="117" alt="image" src="https://user-images.githubusercontent.com/63503783/203756716-43b3ce4b-36bd-4b8a-ab4b-e806cb6e6d4d.png">

```
for dataset in merge:
    dataset['Embarked'] = dataset['Embarked'].map({
        'S': 0,
        'C': 1,
        'Q': 2
    })

train_data.head()
```
<img width="446" alt="image" src="https://user-images.githubusercontent.com/63503783/203756888-38c4937f-7e1b-449d-ae8b-1d60d5c0e280.png">

# 3. Apply Model
We're going to use random forest as the model and calculate the confidence score this time.
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference Wikipedia.

```
X_train = train_data.drop('Survived', axis = 1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis = 1).copy()
#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
model = RandomForestClassifier(n_estimators = 100)

model.fit(X_train, y_train)
```

# 4. Predict And Calculate The Score
Actually, the model has a high score as 98.09, so I suspect there may be a data leak or the way of feature engineering can be improved. For example, the number of passegners for different ages may vary a lot, so I can split ages into groups, instead of remaining it as a continious data. Also, I can add new features like family size, to combine the feature of **Sibsp** and **Parch**, instead of viewing these two features separately. And improvement of data preprocessing and that of the accuracy of the model will on ongoing.

```
predict = model.predict(X_test)
score = round(model.score(X_train, y_train)*100, 2)

score
```
<img width="58" alt="image" src="https://user-images.githubusercontent.com/63503783/203757167-6152a9a6-ada5-495b-9687-f671cea3d6ee.png">
