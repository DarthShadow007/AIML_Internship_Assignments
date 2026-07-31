# Assignment 1: Adult Census Income Classification

## Overview
This project predicts whether income exceeds $50K/yr based on census data using a Random Forest Classifier. It includes a complete pipeline for automated data downloading, missing value imputation, standard scaling, and one-hot encoding.

## Files
* `01_adult_census_income.py`: Main execution script.

## Requirements
```bash
pip install pandas numpy scikit-learn
```

## How to Run
Execute the script from the terminal:
```bash
python 01_adult_census_income.py
```
The script will output the Accuracy, ROC-AUC score, Confusion Matrix, and a detailed Classification Report.