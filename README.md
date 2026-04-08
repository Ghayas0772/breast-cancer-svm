# Breast Cancer Wisconsin Dataset Analysis

This project focuses on **exploratory data analysis (EDA)**, **data cleaning**, and **preprocessing** for the Breast Cancer Wisconsin dataset. The dataset contains 699 observations and 11 features, including a target column (`Class`) that indicates whether a tumor is benign or malignant.

---

## 1. Data Loading & Inspection

```python
import pandas as pd
import numpy as np

# Load dataset
path = "D:\\Eductaional\\NU\\ANA 680\\Modal 1\\Assignment\\Assignment1\\datasets\\breast-cancer-wisconsin.data"
columns = ["ID", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
           "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
           "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]
data = pd.read_csv(path, names=columns, sep=',')

# Check first rows
print(data.head())
print(data.info())

<!--  Handling Missing Values in Bare_Nuclei  -- >
