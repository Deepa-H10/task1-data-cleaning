# 🧹 Task 1 – Data Cleaning and Preprocessing

This repository contains my solution for **Task 1** of the **Data Analyst Internship** program.

## 📌 Objective
Clean and preprocess a raw dataset by handling:
- Missing values
- Duplicates
- Inconsistent data formats
- Categorical encoding
- Numeric scaling  

The goal is to produce a **cleaned dataset** that is ready for analysis or modeling.

---

## 📊 Dataset Used
**Customer Personality Analysis** (from [Kaggle](https://www.kaggle.com/datasets/preetam009/customer-personality-analysis-dataset))  
This dataset contains demographic and purchase behaviour details of customers, including:
- Year of Birth
- Education
- Marital Status
- Number of Children
- Purchase history (MntWines, MntFruits, etc.)
- Campaign responses

---

## 🛠️ Tools & Libraries
- **Python 3.x**
- **pandas** – data cleaning & manipulation  
- **numpy** – numerical operations  
- **scikit-learn** – scaling & encoding  
- **datetime** – date/time feature creation  

---

## 🔧 Data Cleaning Steps

| Step | Action |
|------|--------|
| 1 | **Loaded dataset** and inspected for missing values, datatypes, duplicates |
| 2 | **Dropped columns** with more than 50% missing values |
| 3 | **Imputed missing values** (numeric → median, categorical → mode) |
| 4 | **Removed duplicate rows** |
| 5 | **Converted datatypes**: `Year_Birth` → numeric, `Dt_Customer` → datetime |
| 6 | **Feature Engineering**: created `Age`, `Days_As_Customer`, `Num_Children`, `Total_Spent` |
| 7 | **Encoded categorical variables**: ordinal mapping for Education + one-hot encoding |
| 8 | **Scaled numeric columns** using StandardScaler |
| 9 | **Exported cleaned dataset** as `Customer_Personality_Cleaned.csv` |

---

## 📁 Repository Structure
