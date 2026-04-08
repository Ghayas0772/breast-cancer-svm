# Breast Cancer Classification Project

## Project Overview
This project focuses on building classification models to predict **breast cancer** type (benign or malignant) using the **Breast Cancer Wisconsin dataset**. The analysis includes **data exploration, preprocessing, outlier handling, and correlation analysis** to prepare the dataset for machine learning models such as Logistic Regression, KNN, and SVM.

---

## Dataset
- **Source:** Breast Cancer Wisconsin dataset
- **Number of samples:** 699
- **Number of features:** 11
- **Target variable:** `Class` (2 → benign, 4 → malignant, later mapped to 0 → benign, 1 → malignant)
- **Feature types:**
  - 10 numeric features
  - 1 object feature (`Bare_Nuclei` containing numeric values and some missing entries `?`)

**Columns:**
- ID
- Clump_Thickness
- Uniformity_Cell_Size
- Uniformity_Cell_Shape
- Marginal_Adhesion
- Single_Epithelial_Cell_Size
- Bare_Nuclei
- Bland_Chromatin
- Normal_Nucleoli
- Mitoses
- Class

---

## Data Loading & Inspection
```python
import pandas as pd
import numpy as np

# Load dataset
path = "D:\\Eductaional\\NU\\ANA 680\\Modal 1\\Assignment\\Assignment1\\datasets\\breast-cancer-wisconsin.data"
columns = ["ID", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
           "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
           "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]
data = pd.read_csv(path, names=columns, sep=',')



Check for null values and unique values in Bare_Nuclei:

data['Bare_Nuclei'] = pd.to_numeric(data['Bare_Nuclei'], errors='coerce')  # '?' becomes NaN
data['Bare_Nuclei'].fillna(data['Bare_Nuclei'].median(), inplace=True)

Target variable mapping:

data['Class'] = data['Class'].map({2: 0, 4: 1})  # 0 → Benign, 1 → Malignant
Exploratory Data Analysis (EDA)
1. Feature Distribution
import matplotlib.pyplot as plt
data.drop('ID', axis=1).hist(bins=15, figsize=(15,10))
plt.show()
2. Correlation Analysis
import seaborn as sns
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Correlation with target
data.corrwith(data['Class'])
3. Class Distribution
sns.countplot(x='Class', data=data)
plt.title("Class Distribution: 0 → Benign, 1 → Malignant")
plt.show()

data['Class'].value_counts(normalize=True)
Benign (0): 458 samples (~65%)
Malignant (1): 241 samples (~35%)
Outlier Detection & Handling
numeric_cols = data.select_dtypes(include=np.number).columns

# Cap extreme values (Winsorization)
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = np.where(data[col] < lower, lower, data[col])
    data[col] = np.where(data[col] > upper, upper, data[col])
Total rows with any outlier before capping: 208
Capping preserves dataset size and reduces extreme influence.
Preprocessing Steps
Removed ID column from features
Imputed missing values in Bare_Nuclei with median
Converted Class to 0/1
Removed leading/trailing spaces in column names
Capped extreme outliers in numeric features
X = data.drop(["ID", "Class"], axis=1)
y = data["Class"]
Next Steps
Train classification models: Logistic Regression, KNN, SVM
Evaluate model performance using metrics: accuracy, precision, recall, F1-score
Optional: Feature selection based on correlation analysis
Deploy model using Flask and Docker
Notes
Dataset is slightly imbalanced (~65% benign, 35% malignant)
Outlier handling is critical for ML models sensitive to extreme values
EDA shows high correlation among features: Uniformity_Cell_Size, Uniformity_Cell_Shape, Bare_Nuclei

---

If you want, I can also **add a “Project Setup & Installation” section** with **Flask app and Heroku deployment instructions** so the `README.md` is fully complete for GitHub.  

Do you want me to do that next?

