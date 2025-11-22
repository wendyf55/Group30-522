# Predicting Abalone Ages Based on Individual Characteristics - Regression Problem

## Project Summary

This project builds and compares regression models to predict the age of individual abalones from simple physical measurements. Using the Abalone dataset from the UCI Machine Learning Repository, we model the relationship between age (approximated from shell rings) and features such as sex, shell length, diameter, height, and several weight measurements.

We fit three supervised learning models in Python:

- **Baseline linear regression**
- **Random Forest regressor**
- **Support Vector Regressor (SVR) with an RBF kernel**

Our linear regression model explains about half of the variance in ring count (R² ≈ 0.44, RMSE ≈ 5.48), while the non-linear Random Forest and SVR models both perform better (R² ≈ 0.52, RMSE ≈ 4.76–4.70). Whole weight is a strong positive predictor of age, whereas shucked weight has a strong negative coefficient, suggesting collinearity among weight variables. Overall, these results indicate that flexible, non-linear models using carefully chosen features are better suited for predicting abalone age from physical measurements.

## Contributors

Group 30 in DSCI 522  

- Mehmet Imga  
- Wendy Frankel  
- Claudia Liauw  
- Serene Zha

## How to Run the Analysis

Follow these steps to set up the environment and reproduce our analysis.

### Step 1: Clone the repository

```bash
git clone https://github.com/wendyf55/Group30-522.git
cd Group30-522
```

How to run data analysis: Use the environment.yaml file, run our analysis file.

Dependencies: Pandas, Altair, SciKit Learn, ucimlrepo, ipykernel

Names of the licenses: MIT

n the main README.md file for this project you should include: - the project title - the list of contributors/authors - a short summary of the project (view from 10,000 feet) - how to run your data analysis - a list of the dependencies needed to run your analysis - the names of the licenses contained in LICENSE.md
