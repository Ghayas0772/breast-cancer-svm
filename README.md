# Breast Cancer Wisconsin Dataset Analysis

This project focuses on exploratory data analysis (EDA), data cleaning, and preprocessing for the Breast Cancer Wisconsin dataset. The dataset contains 699 observations and 11 features, including a target column (Class) that indicates whether a tumor is benign or malignant.

### 1. Data Loading & Inspection

```import pandas as pd
import numpy as np```

# Load dataset
```path = "datasets/breast-cancer-wisconsin.data"
columns = ["ID", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
           "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
           "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]
data = pd.read_csv(path, names=columns, sep=',')

# Check first rows
print(data.head())
print(data.info())```


### Handling Missing Values in Bare_Nuclei

```# Convert Bare_Nuclei to numeric; '?' becomes NaN
data['Bare_Nuclei'] = pd.to_numeric(data['Bare_Nuclei'], errors='coerce')

# Fill missing values with median
data['Bare_Nuclei'].fillna(data['Bare_Nuclei'].median(), inplace=True)```


### 2. Target Variable Exploration

```# Map target classes to 0 (Benign) and 1 (Malignant)
data['Class'] = data['Class'].map({2: 0, 4: 1})

# Check distribution
print(data['Class'].value_counts())
print(data['Class'].value_counts(normalize=True))```


### Key findings:

*** 0 → Benign *** 

### 1 → Malignant

The dataset has a slight class imbalance (~65% benign vs 35% malignant).

### 3. Feature Engineering & Preprocessing

#### 3.1 Separate Features & Target
```
X = data.drop(["ID", "Class"], axis=1)
y = data["Class"]
```

3.2 Clean Column Names and Strings
```
# Remove leading/trailing spaces in all column names
data.columns = data.columns.str.strip()

# Remove extra spaces in string columns
string_cols = data.select_dtypes(include="object").columns
data[string_cols] = data[string_cols].apply(lambda x: x.str.strip())
```
4. Exploratory Data Analysis (EDA)

4.1 Feature Distribution & Correlations

import matplotlib.pyplot as plt
import seaborn as sns

# Histograms
data.drop('ID', axis=1).hist(bins=15, figsize=(15,10))
plt.show()
```
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
```

4.2 Correlation Analysis

High correlation features with the target variable (Class):

Uniformity_Cell_Size (~0.82)

Uniformity_Cell_Shape (~0.82)

Bare_Nuclei (~0.82)

5. Outlier Detection & Handling

We use the Interquartile Range (IQR) method to identify extreme values.

```# Identify numeric columns
numeric_cols = data.select_dtypes(include=np.number).columns

# Detect outliers
outlier_indices = []
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_idx = data[(data[col] < lower) | (data[col] > upper)].index
    outlier_indices.extend(outlier_idx)

# Unique outlier rows
outlier_indices = list(set(outlier_indices))
print(f"Total unique rows with any outlier: {len(outlier_indices)}")
```

Outlier Handling: Capping (Winsorization)

To avoid losing data, we cap extreme values at the lower and upper bounds.
```
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Cap values
    data[col] = np.where(data[col] < lower, lower, data[col])
    data[col] = np.where(data[col] > upper, upper, data[col])

```
6. Summary

Dataset Size: 699 observations, 11 features.

Preprocessing: Missing values handled in Bare_Nuclei, and outliers capped.

Readiness: Features are now ready for Machine Learning models like Logistic Regression, SVM, or Random Forest.
