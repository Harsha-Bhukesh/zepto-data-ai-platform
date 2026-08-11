# Module 2 — Analytics Pipeline

## Zepto Data & AI Platform

Module 2 implements an end-to-end analytics and machine-learning workflow using the **Titanic dataset**.

The module is divided into two major parts:

* **Part A — Profiling, Cleaning and Data Story**
* **Part B — Predictive Modeling**

The workflow covers dataset profiling, missing-value treatment, outlier analysis, survival analysis, correlation analysis, multivariate visualization, standardization, classification, class-imbalance handling, hyperparameter tuning, regression, residual analysis, model comparison, and complete model-pipeline persistence.

---

## 1. Objectives

The main objectives of this module are:

1. Profile the Titanic dataset.
2. Report missing-value counts and percentages.
3. Apply percentage-based missing-value handling.
4. Perform IQR-based outlier analysis.
5. Analyze Fare skewness using mean, median and mode.
6. Analyze survival rates by sex, passenger class, and sex plus passenger class.
7. Compute the required correlation matrix.
8. Identify and interpret the two strongest correlations.
9. Create multivariate visualizations with written interpretations.
10. Perform an exploratory standardization check.
11. Perform a stratified train/test split.
12. Build preprocessing pipelines without test-data leakage.
13. Train three classification models.
14. Evaluate the classifiers using the complete metric suite.
15. Compare baseline, balanced class weights and SMOTE.
16. Tune Random Forest using GridSearchCV.
17. Report Random Forest OOB performance.
18. Perform Fare regression.
19. Analyze regression residuals and heteroscedasticity.
20. Compare the final models.
21. Save, reload and test the complete fitted machine-learning pipeline.

---

# 2. Dataset

The module uses the **Titanic dataset**.

The dataset is loaded once during the analytics workflow and an offline CSV fallback is created using:

```python
df.to_csv("titanic.csv", index=False)
```

The committed `titanic.csv` file inside `/analytics` allows the dataset to be loaded offline using:

```python
pd.read_csv("titanic.csv")
```

The raw dataset is not independently reloaded during the modeling stage.

---

# 3. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* imbalanced-learn
* Joblib
* Jupyter Notebook / Google Colab

---

# 4. Module Structure

Recommended structure:

```text
analytics/
│
├── analytics_pipeline.ipynb
├── titanic.csv
├── README.md
│
├── models/
│   └── final_pipeline.joblib
│
└── outputs/
    └── supporting charts / artifacts
```

---

# Part A — Profiling, Cleaning and Data Story

## 5. Dataset Profiling

The original Titanic dataset contains:

* **891 rows**
* **15 columns**

The initial dataset contains the following important missing values:

| Column      | Missing Count | Missing Percentage |
| ----------- | ------------: | -----------------: |
| age         |           177 |           19.8653% |
| embarked    |             2 |            0.2245% |
| deck        |           688 |           77.2166% |
| embark_town |             2 |            0.2245% |

Descriptive statistics were also generated for the numerical variables.

---

## 6. Missing-Value Strategy

A percentage-based rule was used to determine the treatment of missing values:

* **Less than 5% missing:** affected observations are removed.
* **5%–30% missing:** missing values are imputed.
* **Very high missingness:** the column is removed when reliable imputation is not justified.

### Applied decisions

### `age`

Missing percentage:

**19.8653%**

Since this is between 5% and 30%, the missing values were replaced using the **median**.

The median was selected because it is less sensitive to extreme values than the mean.

### `embarked`

Missing percentage:

**0.2245%**

Since the missing percentage is below 5%, the affected observations were removed.

### `embark_town`

Missing percentage:

**0.2245%**

The affected observations were removed together with the missing embarkation records.

### `deck`

Missing percentage:

**77.2166%**

Because more than three quarters of the values were missing, the column was removed rather than performing unreliable imputation.

### Cleaning Result

```text
Original shape: (891, 15)
Cleaned shape: (889, 14)
```

No missing values remained after cleaning.

---

# 7. IQR-Based Outlier Analysis

The Interquartile Range (IQR) method was used to identify potential outliers.

The IQR boundaries were calculated using:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

### Age

```text
Age outliers: 65
Lower bound: 2.5
Upper bound: 54.5
```

### Fare

```text
Fare outliers: 114
Lower bound: -26.7605
Upper bound: 65.6563
```

The detected values were retained because they can represent legitimate passenger characteristics rather than data-entry errors. In particular, high Fare values can represent expensive tickets.

---

# 8. Fare Skewness Analysis

The calculated Fare statistics were:

```text
Mean:   32.0967
Median: 14.4542
Mode:    8.05
Skewness: 4.8014
```

The relationship:

```text
Mean > Median > Mode
```

combined with the strongly positive skewness value indicates that **Fare is highly right-skewed**.

The long right tail is caused by a relatively small number of passengers paying substantially higher fares.

---

# 9. Survival Rate Analysis

## Survival by Sex

The calculated survival rates were:

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |      0.740385 |
| Male   |      0.188908 |

Female passengers had a substantially higher survival rate than male passengers.

---

## Survival by Passenger Class

| Passenger Class | Survival Rate |
| --------------- | ------------: |
| 1               |      0.626168 |
| 2               |      0.472826 |
| 3               |      0.242363 |

First-class passengers had the highest survival rate, while third-class passengers had the lowest.

---

## Survival by Sex and Passenger Class

| Sex    | Class | Survival Rate |
| ------ | ----: | ------------: |
| Female |     1 |      0.967391 |
| Female |     2 |      0.921053 |
| Female |     3 |      0.500000 |
| Male   |     1 |      0.368852 |
| Male   |     2 |      0.157407 |
| Male   |     3 |      0.135447 |

The combined analysis shows that both sex and passenger class were strongly associated with survival.

Female passengers in first and second class had particularly high survival rates, while third-class male passengers had the lowest survival rate.

---

# 10. Correlation Analysis

The correlation matrix was calculated using **exactly the six required columns**:

```text
survived
pclass
age
sibsp
parch
fare
```

The columns `adult_male` and `alone` were excluded.

### Strongest Correlations

The two strongest absolute off-diagonal correlations were:

### 1. `pclass` and `fare`

```text
Correlation = -0.548193
```

The negative relationship occurs because smaller numerical values of `pclass` represent higher passenger classes, which generally had higher fares.

### 2. `parch` and `sibsp`

```text
Correlation = 0.414542
```

The positive relationship indicates that passengers traveling with more parents/children also tended to travel with more siblings/spouses, reflecting family-group travel.

The correlation matrix was also visualized using a heatmap.

---

# 11. Multivariate Visualization

At least four multivariate charts were created.

### Chart 1 — Survival Rate by Passenger Class and Sex

This chart demonstrates that survival varied substantially across both sex and passenger class.

Female passengers generally had higher survival rates, while male passengers, especially in third class, had much lower survival rates.

### Chart 2 — Age Distribution by Sex

The age distribution shows the range of passenger ages for male and female passengers and demonstrates that the dataset contains children, adults and older passengers.

### Chart 3 — Fare Distribution by Class and Survival

Fare varies substantially between passenger classes. First-class passengers generally paid higher fares, and Fare also shows differences between survivors and non-survivors.

### Chart 4 — Survival Rate by Family Size and Sex

Family size was calculated as:

```text
family_size = sibsp + parch + 1
```

Survival varies across family sizes, with females generally showing higher survival rates. Very large family groups contain fewer observations and should therefore be interpreted cautiously.

---

# 12. Standardization Check

An exploratory standardization check was performed on:

* `age`
* `fare`

The transformation used was:

```text
z = (x - mean) / standard deviation
```

### Before Standardization

```text
Age mean: 29.315151856017994
Age std: 12.984932293690774

Fare mean: 32.09668087739032
Fare std: 49.69750431670801
```

### After Standardization

```text
Age mean: approximately 0
Age std: approximately 1

Fare mean: approximately 0
Fare std: approximately 1
```

This confirms that standardization successfully places both variables on a common scale.

The modeling pipeline performs its own training-only preprocessing and scaling.

---

# Part B — Predictive Modeling

## 13. Stratified Train/Test Split

The cleaned dataset was divided into training and testing sets using a **stratified split**.

```text
X_train: (711, 14)
X_test:  (178, 14)

y_train: (711,)
y_test:  (178,)
```

### Training Distribution

```text
0    0.61744
1    0.38256
```

### Testing Distribution

```text
0    0.617978
1    0.382022
```

Stratification preserves approximately the same target-class proportions in both datasets and therefore provides a fair evaluation of the classifiers.

The split was performed **before modeling preprocessing**.

---

# 14. Preprocessing

The numerical features were:

```text
pclass
age
sibsp
parch
fare
family_size
```

The categorical features were:

```text
sex
embarked
```

### Numerical preprocessing

1. Median imputation
2. StandardScaler

### Categorical preprocessing

1. Most-frequent imputation
2. One-hot encoding

The preprocessing was implemented using a `ColumnTransformer` and scikit-learn `Pipeline`.

This ensures that preprocessing parameters are learned from the training data and then applied to the test data without fitting preprocessing on the test set.

---

# 15. Classification Models

Three classifiers were trained using the identical train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

# 16. Initial Classification Results

## Logistic Regression

```text
Accuracy:  0.808989
Precision: 0.783333
Recall:    0.691176
F1:        0.734375
AUC:       0.860963
```

Confusion Matrix:

```text
[[97 13]
 [21 47]]
```

---

## Decision Tree

```text
Accuracy:  0.758427
Precision: 0.681159
Recall:    0.691176
F1:        0.686131
AUC:       0.737834
```

Confusion Matrix:

```text
[[88 22]
 [21 47]]
```

---

## Random Forest

```text
Accuracy:  0.786517
Precision: 0.727273
Recall:    0.705882
F1:        0.716418
AUC:       0.820455
```

Confusion Matrix:

```text
[[92 18]
 [20 48]]
```

Logistic Regression achieved the highest initial AUC and accuracy among the three initial classifiers.

---

# 17. Decision Tree Visualization

The Decision Tree was visualized using `plot_tree`.

The visualization includes:

* Feature names
* Class names
* Tree structure
* Decision splits

The visualization demonstrates how the model recursively separates passengers into survival classes.

---

# 18. Class Imbalance Comparison

Three approaches were compared:

1. Baseline
2. `class_weight="balanced"`
3. SMOTE

### Results

| Strategy                  | Precision |   Recall |       F1 |
| ------------------------- | --------: | -------: | -------: |
| Baseline                  |  0.783333 | 0.691176 | 0.734375 |
| `class_weight='balanced'` |  0.718310 | 0.750000 | 0.733813 |
| SMOTE                     |  0.724638 | 0.735294 | 0.729927 |

### Training Distribution

Original:

```text
0    439
1    272
```

After SMOTE:

```text
0    439
1    439
```

SMOTE was applied **only to the training fold**.

### Conclusion

The baseline model achieved the highest precision and F1 score.

The balanced class-weight model achieved the highest recall.

SMOTE also improved recall compared with the baseline but produced a slightly lower F1 score.

Therefore, the baseline provides the strongest overall balance, while class weighting is useful when recall is the primary objective.

---

# 19. Random Forest Hyperparameter Tuning

`GridSearchCV` was used to tune the Random Forest.

### Best Parameters

```text
max_depth = 5
max_features = 'sqrt'
n_estimators = 200
```

### Best Cross-Validation F1

```text
0.755718
```

### OOB Score

```text
0.822785
```

The Random Forest was configured with:

```python
oob_score=True
```

The tuning results indicate improved generalization compared with the untuned Random Forest.

---

# 20. Tuned Random Forest Results

The tuned Random Forest achieved:

```text
Accuracy:  0.820225
Precision: 0.821429
Recall:    0.676471
F1:        0.741935
AUC:       0.839840
```

Confusion Matrix:

```text
[[100  10]
 [ 22  46]]
```

A ROC curve was also generated for the tuned Random Forest.

The tuned Random Forest improved accuracy, precision and F1 compared with the original Random Forest.

---

# 21. Fare Regression

A multivariate Linear Regression model was created to predict `fare`.

The regression task reports all four required metrics:

```text
MAE:          18.3735
RMSE:         41.2921
R²:            0.3609
Adjusted R²:   0.2558
```

The R² value indicates that approximately 36.1% of the variation in Fare is explained by the available predictors.

Therefore, the regression model has limited explanatory power and should not be considered the primary predictive model.

---

# 22. Residual Analysis

Residuals were calculated as:

```text
Residual = Actual Fare - Predicted Fare
```

Results:

```text
Residual Mean:
-0.588844

Residual Standard Deviation:
41.404394
```

A residual plot was created to examine the relationship between predicted Fare and residuals.

### Heteroscedasticity Conclusion

The residual spread is not perfectly constant across the range of predicted Fare values.

Higher predicted Fare values show larger residual variability in parts of the residual plot.

Therefore, there is evidence of **heteroscedasticity**, meaning the constant-variance assumption of ordinary linear regression is not fully satisfied.

---

# 23. Final Classification Comparison

| Model               | Accuracy | Precision |   Recall |       F1 |      AUC |
| ------------------- | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression | 0.808989 |  0.783333 | 0.691176 | 0.734375 | 0.860963 |
| Decision Tree       | 0.758427 |  0.681159 | 0.691176 | 0.686131 | 0.737834 |
| Random Forest       | 0.786517 |  0.727273 | 0.705882 | 0.716418 | 0.820455 |
| Tuned Random Forest | 0.820225 |  0.821429 | 0.676471 | 0.741935 | 0.839840 |

The tuned Random Forest achieved the strongest overall combination of accuracy, precision and F1 score.

Logistic Regression achieved the highest AUC.

---

# 24. Final Recommendation

The **Tuned Random Forest** is selected as the final classification model.

Reasons:

* Accuracy: **82.02%**
* Precision: **82.14%**
* F1: **74.19%**
* AUC: **83.98%**
* OOB Score: **82.28%**
* Best cross-validation F1: **75.57%**

Although Logistic Regression achieved a higher AUC of approximately 86.10%, the tuned Random Forest provided stronger accuracy, precision and F1 performance.

Therefore, the tuned Random Forest is the preferred final classifier for this module.

The Fare regression model is treated separately because it is a regression task and its R² of approximately 36.09% indicates limited predictive explanatory power.

---

# 25. Model Persistence

The complete fitted classification pipeline is saved using Joblib:

```python
joblib.dump(full_pipeline, "final_pipeline.joblib")
```

The saved artifact contains:

* preprocessing
* imputation
* encoding
* scaling
* final estimator

The complete pipeline can therefore receive raw new data and perform preprocessing and prediction end-to-end.

The saved pipeline is also reloaded using:

```python
loaded_pipeline = joblib.load("final_pipeline.joblib")
```

A raw test sample is passed through the reloaded pipeline to demonstrate successful end-to-end inference.

---

# 26. Running the Module

## Google Colab

Open:

```text
analytics_pipeline.ipynb
```

Run the notebook cells sequentially from top to bottom.

The notebook contains the executed outputs, tables, visualizations, model metrics and interpretations required for the module.

## Offline Dataset

The committed Titanic fallback dataset can be loaded using:

```python
import pandas as pd

df = pd.read_csv("titanic.csv")
```

---

# 27. Submission Contents

The `/analytics` directory contains the Module 2 artifacts:

```text
analytics/
│
├── analytics_pipeline.ipynb
├── titanic.csv
├── README.md
│
├── models/
│   └── final_pipeline.joblib
│
└── outputs/
    └── supporting charts
```

The notebook contains the complete executed EDA and modeling workflow, including the required outputs and visualizations.

---

# 28. Acceptance Criteria Checklist

| Requirement                                        | Status |
| -------------------------------------------------- | ------ |
| Missing-value percentages reported                 | ✅      |
| Percentage-based missing-value strategy documented | ✅      |
| `titanic.csv` offline fallback                     | ✅      |
| Dataset loaded once for the module                 | ✅      |
| IQR outlier counts for Age and Fare                | ✅      |
| Fare mean, median, mode and skewness               | ✅      |
| Survival by sex                                    | ✅      |
| Survival by passenger class                        | ✅      |
| Survival by sex + passenger class                  | ✅      |
| Correlation matrix on required six columns         | ✅      |
| `adult_male` and `alone` excluded                  | ✅      |
| Two strongest correlations interpreted             | ✅      |
| At least four multivariate charts                  | ✅      |
| Written chart interpretations                      | ✅      |
| Before/after standardization                       | ✅      |
| Stratified train/test split                        | ✅      |
| Training-only preprocessing                        | ✅      |
| Three classifiers                                  | ✅      |
| Confusion matrix and full metrics                  | ✅      |
| Decision Tree visualization                        | ✅      |
| Baseline vs balanced vs SMOTE                      | ✅      |
| SMOTE training fold only                           | ✅      |
| GridSearchCV                                       | ✅      |
| Best Random Forest parameters                      | ✅      |
| OOB score                                          | ✅      |
| Regression MAE, RMSE, R², Adjusted R²              | ✅      |
| Heteroscedasticity conclusion                      | ✅      |
| Final model comparison                             | ✅      |
| Classifier/regression metrics separated            | ✅      |
| Complete fitted pipeline saved                     | ✅      |
| Pipeline reload demonstrated                       | ✅      |
| End-to-end raw-data prediction demonstrated        | ✅      |

---

## Final Outcome

Module 2 provides a complete analytics and predictive-modeling workflow for the Titanic dataset, progressing from data profiling and exploratory analysis to validated machine-learning models and a persisted end-to-end prediction pipeline.

**Final recommended classifier: Tuned Random Forest**

