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

