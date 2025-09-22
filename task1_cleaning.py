"""
Task 1: Data Cleaning and Preprocessing
Dataset: Customer Personality Analysis (from Kaggle)
Author: Deepa Basappa Hugar
Description: This script loads, cleans, and preprocesses the dataset,
and outputs a cleaned CSV file ready for analysis or modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler


# 1. Load Dataset

df = pd.read_csv('Customer_Personality_Analysis_Dataset.csv')  # <-- update path if needed
print("✅ Dataset loaded successfully!")
print("Shape before cleaning:", df.shape)


# 2. Initial Inspection

print("\n🔎 Data Info:")
print(df.info())

print("\nMissing Values per Column:")
print(df.isnull().sum())


# 3. Handle Missing Values

# Drop columns with > 50% missing values
threshold = 0.5 * len(df)
cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > threshold]
df.drop(columns=cols_to_drop, inplace=True)
print(f"\n🧹 Dropped columns with >50% missing values: {cols_to_drop}")

# Fill numeric missing values with median, categorical with mode
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)


# 4. Remove Duplicates

duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"\n🗑 Removed {duplicates} duplicate rows")


# 5. Fix Data Types

# Convert Year_Birth to numeric safely
if 'Year_Birth' in df.columns:
    df['Year_Birth'] = pd.to_numeric(df['Year_Birth'], errors='coerce')
    df['Year_Birth'].fillna(df['Year_Birth'].median(), inplace=True)

    current_year = datetime.now().year
    df['Age'] = current_year - df['Year_Birth']

# Convert Dt_Customer to datetime if exists
if 'Dt_Customer' in df.columns:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    today = pd.to_datetime('today')
    df['Days_As_Customer'] = (today - df['Dt_Customer']).dt.days


# 6. Feature Engineering

# Number of Children
if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
    df['Kidhome'] = pd.to_numeric(df['Kidhome'], errors='coerce').fillna(0)
    df['Teenhome'] = pd.to_numeric(df['Teenhome'], errors='coerce').fillna(0)
    df['Num_Children'] = df['Kidhome'] + df['Teenhome']

# Total Amount Spent
purchase_cols = [c for c in df.columns if c.startswith('Mnt')]
if purchase_cols:
    for col in purchase_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Total_Spent'] = df[purchase_cols].sum(axis=1)


# 7. Encode Categorical Variables

# Ordinal Encoding for Education if present
edu_map = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
if 'Education' in df.columns:
    df['Education_Num'] = df['Education'].map(edu_map)
    df['Education_Num'].fillna(df['Education_Num'].median(), inplace=True)

# One-hot encode remaining categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# 8. Scaling Numeric Columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:  # Avoid error if no numeric cols
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 9. Save Cleaned Dataset
df.to_csv('Customer_Personality_Cleaned.csv', index=False)
print("\n✅ Cleaning completed successfully!")
print("Shape after cleaning:", df.shape)
print("Cleaned dataset saved as 'Customer_Personality_Cleaned.csv'")
