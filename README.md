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

Add-ons:
EDA
Pipeline
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
