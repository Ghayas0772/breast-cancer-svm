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

### Check for null values and unique values in Bare_Nuclei:
data['Bare_Nuclei'] = pd.to_numeric(data['Bare_Nuclei'], errors='coerce')  # '?' becomes NaN
data['Bare_Nuclei'].fillna(data['Bare_Nuclei'].median(), inplace=True)
