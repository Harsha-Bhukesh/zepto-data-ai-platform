# ============================================================
# MODULE 2 — ANALYTICS PIPELINE
# ZEPTO DATA & AI PLATFORM
# ============================================================


# ============================================================
# STEP 1 — IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE


# ============================================================
# PART A — PROFILING, CLEANING AND DATA STORY
# ============================================================


# ============================================================
# STEP 2 — LOAD TITANIC DATASET
# ============================================================

df = sns.load_dataset("titanic")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# STEP 3 — SAVE OFFLINE TITANIC DATASET
# ============================================================

df.to_csv("titanic.csv", index=False)

print("titanic.csv saved successfully!")


# ============================================================
# STEP 4 — DATASET INFORMATION
# ============================================================

print("========== DATASET INFO ==========")
df.info()

print("\n========== DESCRIPTIVE STATISTICS ==========")
display(df.describe())

print("\n========== DATASET SHAPE ==========")
print(df.shape)


# ============================================================
# STEP 5 — MISSING VALUE ANALYSIS
# ============================================================

missing_count = df.isnull().sum()
missing_percentage = df.isnull().mean() * 100

missing_df = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage
})

missing_df = missing_df[
    missing_df["Missing_Count"] > 0
]

print("========== MISSING VALUES ==========")
display(missing_df)


# ============================================================
# STEP 6 — MISSING VALUE CLEANING
# ============================================================

df_clean = df.copy()

df_clean = df_clean.dropna(
    subset=["embarked", "embark_town"]
)

df_clean["age"] = df_clean["age"].fillna(
    df_clean["age"].median()
)

df_clean = df_clean.drop(
    columns=["deck"]
)

print("Cleaning completed!")
print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)

print("\nRemaining missing values:")
print(df_clean.isnull().sum().sum())


# ============================================================
# STEP 7 — AGE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# ============================================================
# STEP 8 — AGE BOX PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df_clean["age"]
)

plt.title("Age Box Plot")
plt.xlabel("Age")
plt.show()


# ============================================================
# STEP 9 — FARE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["fare"],
    bins=30,
    kde=True
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()


# ============================================================
# STEP 10 — FARE BOX PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df_clean["fare"]
)

plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.show()


# ============================================================
# STEP 11 — IQR OUTLIER ANALYSIS
# ============================================================

def iqr_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[
        (series < lower) |
        (series > upper)
    ]

    return len(outliers), lower, upper


age_outliers, age_lower, age_upper = iqr_outliers(
    df_clean["age"]
)

fare_outliers, fare_lower, fare_upper = iqr_outliers(
    df_clean["fare"]
)

print("Age outliers:", age_outliers)
print("Age lower bound:", age_lower)
print("Age upper bound:", age_upper)

print("\nFare outliers:", fare_outliers)
print("Fare lower bound:", fare_lower)
print("Fare upper bound:", fare_upper)


# ============================================================
# STEP 12 — FARE STATISTICS AND SKEWNESS
# ============================================================

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
fare_skewness = df_clean["fare"].skew()

print("Fare Mean:", fare_mean)
print("Fare Median:", fare_median)
print("Fare Mode:", fare_mode)
print("Fare Skewness:", fare_skewness)


# ============================================================
# STEP 13 — SURVIVAL RATE BY SEX
# ============================================================

survival_by_sex = (
    df_clean
    .groupby("sex")["survived"]
    .mean()
)

print("Survival Rate by Sex")
display(survival_by_sex)


# ============================================================
# STEP 14 — SURVIVAL RATE BY PASSENGER CLASS
# ============================================================

survival_by_class = (
    df_clean
    .groupby("pclass")["survived"]
    .mean()
)

print("Survival Rate by Passenger Class")
display(survival_by_class)


# ============================================================
# STEP 15 — SURVIVAL RATE BY SEX AND PASSENGER CLASS
# ============================================================

survival_by_sex_class = (
    df_clean
    .groupby(
        ["sex", "pclass"]
    )["survived"]
    .mean()
    .reset_index()
)

print("Survival Rate by Sex and Passenger Class")
display(survival_by_sex_class)


# ============================================================
# STEP 16 — CORRELATION MATRIX
# ============================================================

corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df_clean[
    corr_cols
].corr()

display(corr_matrix)


# ============================================================
# STEP 17 — CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 18 — TWO STRONGEST CORRELATIONS
# ============================================================

corr_pairs = corr_matrix.where(
    ~np.eye(
        corr_matrix.shape[0],
        dtype=bool
    )
).stack()

corr_pairs = corr_pairs[
    corr_pairs.index.map(
        lambda x: x[0] < x[1]
    )
]

corr_pairs = corr_pairs.reindex(
    corr_pairs.abs()
    .sort_values(
        ascending=False
    )
    .index
)

print("Strongest correlations:")
display(
    corr_pairs.head(5)
)


# ============================================================
# STEP 19 — CREATE FAMILY SIZE
# ============================================================

df_clean["family_size"] = (
    df_clean["sibsp"] +
    df_clean["parch"] +
    1
)


# ============================================================
# STEP 20 — MULTIVARIATE CHART 1
# SURVIVAL BY CLASS AND SEX
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Passenger Class and Sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


# ============================================================
# STEP 21 — MULTIVARIATE CHART 2
# AGE DISTRIBUTION BY SEX
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df_clean,
    x="age",
    hue="sex",
    bins=20,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Age Distribution by Sex")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()


# ============================================================
# STEP 22 — MULTIVARIATE CHART 3
# FARE BY CLASS AND SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title(
    "Fare Distribution by Class and Survival"
)

plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()


# ============================================================
# STEP 23 — MULTIVARIATE CHART 4
# FAMILY SIZE AND SEX
# ============================================================

plot_df = df_clean[
    df_clean["family_size"] <= 7
]

plt.figure(figsize=(9, 5))

sns.barplot(
    data=plot_df,
    x="family_size",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Family Size and Sex"
)

plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.show()


# ============================================================
# STEP 24 — STANDARDIZATION CHECK
# ============================================================

scaler = StandardScaler()

standardized = scaler.fit_transform(
    df_clean[
        ["age", "fare"]
    ]
)

df_standardized = df_clean.copy()

df_standardized[
    ["age", "fare"]
] = standardized

print("Before standardization:")

print(
    "Age mean:",
    df_clean["age"].mean()
)

print(
    "Age std:",
    df_clean["age"].std()
)

print(
    "Fare mean:",
    df_clean["fare"].mean()
)

print(
    "Fare std:",
    df_clean["fare"].std()
)

print("\nAfter standardization:")

print(
    "Age mean:",
    df_standardized["age"].mean()
)

print(
    "Age std:",
    df_standardized["age"].std()
)

print(
    "Fare mean:",
    df_standardized["fare"].mean()
)

print(
    "Fare std:",
    df_standardized["fare"].std()
)


# ============================================================
# PART B — PREDICTIVE MODELING
# ============================================================


# ============================================================
# STEP 25 — STRATIFIED TRAIN/TEST SPLIT
# ============================================================

X = df_clean.drop(
    columns=["survived"]
)

y = df_clean["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "X_train shape:",
    X_train.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "y_test shape:",
    y_test.shape
)

print("\nTraining class distribution:")

print(
    y_train.value_counts(
        normalize=True
    )
)

print("\nTesting class distribution:")

print(
    y_test.value_counts(
        normalize=True
    )
)


# ============================================================
# STEP 26 — PREPROCESSING PIPELINE
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])

print(
    "Preprocessor created successfully!"
)

print(
    "Numeric features:",
    numeric_features
)

print(
    "Categorical features:",
    categorical_features
)


# ============================================================
# STEP 27 — TRAIN THREE CLASSIFIERS
# ============================================================

logistic_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

tree_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        DecisionTreeClassifier(
            random_state=42
        )
    )
])

forest_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            random_state=42
        )
    )
])

logistic_pipeline.fit(
    X_train,
    y_train
)

tree_pipeline.fit(
    X_train,
    y_train
)

forest_pipeline.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression trained successfully!"
)

print(
    "Decision Tree trained successfully!"
)

print(
    "Random Forest trained successfully!"
)


# ============================================================
# STEP 28 — EVALUATE THREE CLASSIFIERS
# ============================================================

models = {
    "Logistic Regression":
        logistic_pipeline,

    "Decision Tree":
        tree_pipeline,

    "Random Forest":
        forest_pipeline
}

results = []

roc_data = {}

for name, model in models.items():

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_data[name] = (
        fpr,
        tpr,
        auc
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    })

    print(
        f"\n{name}"
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    print(
        "Accuracy:",
        accuracy
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1:",
        f1
    )

    print(
        "AUC:",
        auc
    )

results_df = pd.DataFrame(
    results
)

print(
    "\nModel Comparison:"
)

display(
    results_df
)


# ============================================================
# STEP 29 — ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

for name, (
    fpr,
    tpr,
    auc
) in roc_data.items():

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves for Classification Models"
)

plt.legend()

plt.show()


# ============================================================
# STEP 30 — DECISION TREE VISUALIZATION
# ============================================================

tree_model = (
    tree_pipeline
    .named_steps["classifier"]
)

tree_preprocessor = (
    tree_pipeline
    .named_steps["preprocessor"]
)

feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    max_depth=3,
    fontsize=9
)

plt.title(
    "Decision Tree Classifier"
)

plt.show()


# ============================================================
# STEP 31 — CLASS IMBALANCE COMPARISON
# ============================================================

baseline_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

balanced_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

baseline_model.fit(
    X_train,
    y_train
)

balanced_model.fit(
    X_train,
    y_train
)

y_pred_baseline = (
    baseline_model.predict(
        X_test
    )
)

y_pred_balanced = (
    balanced_model.predict(
        X_test
    )
)

smote_preprocessor = (
    preprocessor.fit(
        X_train,
        y_train
    )
)

X_train_processed = (
    smote_preprocessor.transform(
        X_train
    )
)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_processed,
        y_train
    )
)

smote_classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

smote_classifier.fit(
    X_train_smote,
    y_train_smote
)

X_test_processed = (
    smote_preprocessor.transform(
        X_test
    )
)

y_pred_smote = (
    smote_classifier.predict(
        X_test_processed
    )
)

imbalance_results = pd.DataFrame([
    {
        "Strategy":
            "Baseline",

        "Precision":
            precision_score(
                y_test,
                y_pred_baseline
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred_baseline
            ),

        "F1":
            f1_score(
                y_test,
                y_pred_baseline
            )
    },

    {
        "Strategy":
            "class_weight='balanced'",

        "Precision":
            precision_score(
                y_test,
                y_pred_balanced
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred_balanced
            ),

        "F1":
            f1_score(
                y_test,
                y_pred_balanced
            )
    },

    {
        "Strategy":
            "SMOTE",

        "Precision":
            precision_score(
                y_test,
                y_pred_smote
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred_smote
            ),

        "F1":
            f1_score(
                y_test,
                y_pred_smote
            )
    }
])

display(
    imbalance_results
)

print(
    "Original training distribution:"
)

print(
    y_train.value_counts()
)

print(
    "\nSMOTE training distribution:"
)

print(
    pd.Series(
        y_train_smote
    ).value_counts()
)


# ============================================================
# STEP 32 — RANDOM FOREST GRIDSEARCHCV
# ============================================================

rf_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            oob_score=True
        )
    )
])

param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        None,
        5,
        10
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_rf_pipeline = (
    grid_search.best_estimator_
)

best_rf = (
    best_rf_pipeline
    .named_steps["classifier"]
)

print(
    "Best Parameters:"
)

print(
    grid_search.best_params_
)

print(
    "\nBest Cross-Validation F1:"
)

print(
    grid_search.best_score_
)

print(
    "\nOOB Score:"
)

print(
    best_rf.oob_score_
)


# ============================================================
# STEP 33 — TUNED RANDOM FOREST EVALUATION
# ============================================================

y_pred_tuned_rf = (
    best_rf_pipeline
    .predict(X_test)
)

y_prob_tuned_rf = (
    best_rf_pipeline
    .predict_proba(
        X_test
    )[:, 1]
)

tuned_rf_cm = (
    confusion_matrix(
        y_test,
        y_pred_tuned_rf
    )
)

tuned_rf_accuracy = (
    accuracy_score(
        y_test,
        y_pred_tuned_rf
    )
)

tuned_rf_precision = (
    precision_score(
        y_test,
        y_pred_tuned_rf
    )
)

tuned_rf_recall = (
    recall_score(
        y_test,
        y_pred_tuned_rf
    )
)

tuned_rf_f1 = (
    f1_score(
        y_test,
        y_pred_tuned_rf
    )
)

tuned_rf_auc = (
    roc_auc_score(
        y_test,
        y_prob_tuned_rf
    )
)

print(
    "Tuned Random Forest"
)

print(
    "Confusion Matrix:"
)

print(
    tuned_rf_cm
)

print(
    "Accuracy:",
    tuned_rf_accuracy
)

print(
    "Precision:",
    tuned_rf_precision
)

print(
    "Recall:",
    tuned_rf_recall
)

print(
    "F1:",
    tuned_rf_f1
)

print(
    "AUC:",
    tuned_rf_auc
)


# ============================================================
# STEP 34 — TUNED RANDOM FOREST ROC CURVE
# ============================================================

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_tuned_rf
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Tuned Random Forest (AUC = {tuned_rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Tuned Random Forest ROC Curve"
)

plt.legend()

plt.show()


# ============================================================
# STEP 35 — FARE REGRESSION
# ============================================================

regression_df = df_clean.drop(
    columns=["fare"]
)

X_reg = regression_df.drop(
    columns=["survived"],
    errors="ignore"
)

y_reg = df_clean["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

reg_numeric_features = (
    X_reg_train
    .select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    )
    .columns
    .tolist()
)

reg_categorical_features = (
    X_reg_train
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)

reg_numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])

reg_categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

reg_preprocessor = ColumnTransformer([
    (
        "num",
        reg_numeric_pipeline,
        reg_numeric_features
    ),
    (
        "cat",
        reg_categorical_pipeline,
        reg_categorical_features
    )
])

regression_pipeline = Pipeline([
    (
        "preprocessor",
        reg_preprocessor
    ),
    (
        "regressor",
        LinearRegression()
    )
])

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

y_reg_pred = (
    regression_pipeline.predict(
        X_reg_test
    )
)

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

n = len(
    y_reg_test
)

p = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(
        X_reg_test
    )
    .shape[1]
)

adjusted_r2 = 1 - (
    ((1 - r2) * (n - 1))
    /
    (n - p - 1)
)

print(
    "MAE:",
    mae
)

print(
    "RMSE:",
    rmse
)

print(
    "R²:",
    r2
)

print(
    "Adjusted R²:",
    adjusted_r2
)


# ============================================================
# STEP 36 — REGRESSION RESIDUAL PLOT
# ============================================================

residuals = (
    y_reg_test -
    y_reg_pred
)

plt.figure(figsize=(8, 5))

plt.scatter(
    y_reg_pred,
    residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Residual Plot - Fare Regression"
)

plt.show()

print(
    "Residual Mean:",
    residuals.mean()
)

print(
    "Residual Standard Deviation:",
    residuals.std()
)


# ============================================================
# STEP 37 — FINAL CLASSIFICATION COMPARISON
# ============================================================

final_classification = pd.DataFrame([
    {
        "Model":
            "Logistic Regression",

        "Accuracy":
            accuracy_score(
                y_test,
                logistic_pipeline.predict(
                    X_test
                )
            ),

        "Precision":
            precision_score(
                y_test,
                logistic_pipeline.predict(
                    X_test
                )
            ),

        "Recall":
            recall_score(
                y_test,
                logistic_pipeline.predict(
                    X_test
                )
            ),

        "F1":
            f1_score(
                y_test,
                logistic_pipeline.predict(
                    X_test
                )
            ),

        "AUC":
            roc_auc_score(
                y_test,
                logistic_pipeline.predict_proba(
                    X_test
                )[:, 1]
            )
    },

    {
        "Model":
            "Decision Tree",

        "Accuracy":
            accuracy_score(
                y_test,
                tree_pipeline.predict(
                    X_test
                )
            ),

        "Precision":
            precision_score(
                y_test,
                tree_pipeline.predict(
                    X_test
                )
            ),

        "Recall":
            recall_score(
                y_test,
                tree_pipeline.predict(
                    X_test
                )
            ),

        "F1":
            f1_score(
                y_test,
                tree_pipeline.predict(
                    X_test
                )
            ),

        "AUC":
            roc_auc_score(
                y_test,
                tree_pipeline.predict_proba(
                    X_test
                )[:, 1]
            )
    },

    {
        "Model":
            "Random Forest",

        "Accuracy":
            accuracy_score(
                y_test,
                forest_pipeline.predict(
                    X_test
                )
            ),

        "Precision":
            precision_score(
                y_test,
                forest_pipeline.predict(
                    X_test
                )
            ),

        "Recall":
            recall_score(
                y_test,
                forest_pipeline.predict(
                    X_test
                )
            ),

        "F1":
            f1_score(
                y_test,
                forest_pipeline.predict(
                    X_test
                )
            ),

        "AUC":
            roc_auc_score(
                y_test,
                forest_pipeline.predict_proba(
                    X_test
                )[:, 1]
            )
    },

    {
        "Model":
            "Tuned Random Forest",

        "Accuracy":
            tuned_rf_accuracy,

        "Precision":
            tuned_rf_precision,

        "Recall":
            tuned_rf_recall,

        "F1":
            tuned_rf_f1,

        "AUC":
            tuned_rf_auc
    }
])

display(
    final_classification.round(4)
)


# ============================================================
# STEP 38 — FINAL MODEL COMPARISON
# ============================================================

final_model_comparison = pd.DataFrame([
    {
        "Model":
            "Logistic Regression",

        "Type":
            "Classification",

        "Accuracy":
            0.808989,

        "Precision":
            0.783333,

        "Recall":
            0.691176,

        "F1":
            0.734375,

        "AUC":
            0.860963,

        "MAE":
            np.nan,

        "RMSE":
            np.nan,

        "R2":
            np.nan,

        "Adjusted_R2":
            np.nan
    },

    {
        "Model":
            "Decision Tree",

        "Type":
            "Classification",

        "Accuracy":
            0.758427,

        "Precision":
            0.681159,

        "Recall":
            0.691176,

        "F1":
            0.686131,

        "AUC":
            0.737834,

        "MAE":
            np.nan,

        "RMSE":
            np.nan,

        "R2":
            np.nan,

        "Adjusted_R2":
            np.nan
    },

    {
        "Model":
            "Tuned Random Forest",

        "Type":
            "Classification",

        "Accuracy":
            0.820225,

        "Precision":
            0.821429,

        "Recall":
            0.676471,

        "F1":
            0.741935,

        "AUC":
            0.839840,

        "MAE":
            np.nan,

        "RMSE":
            np.nan,

        "R2":
            np.nan,

        "Adjusted_R2":
            np.nan
    },

    {
        "Model":
            "Multivariate Linear Regression",

        "Type":
            "Regression",

        "Accuracy":
            np.nan,

        "Precision":
            np.nan,

        "Recall":
            np.nan,

        "F1":
            np.nan,

        "AUC":
            np.nan,

        "MAE":
            18.373542,

        "RMSE":
            41.292124,

        "R2":
            0.360916,

        "Adjusted_R2":
            0.255804
    }
])

display(
    final_model_comparison.round(4)
)


# ============================================================
# STEP 39 — SAVE COMPLETE FINAL PIPELINE
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = (
    "models/final_pipeline.joblib"
)

joblib.dump(
    best_rf_pipeline,
    model_path
)

print(
    "Final pipeline saved successfully!"
)

print(
    "Path:",
    model_path
)


# ============================================================
# STEP 40 — RELOAD AND TEST COMPLETE PIPELINE
# ============================================================

loaded_pipeline = joblib.load(
    "models/final_pipeline.joblib"
)

raw_sample = X_test.iloc[:5].copy()

predictions = (
    loaded_pipeline.predict(
        raw_sample
    )
)

print(
    "Pipeline reloaded successfully!"
)

print(
    "Predictions:",
    predictions
)

print(
    "Raw input shape:",
    raw_sample.shape
)


# ============================================================
# MODULE 2 — COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MODULE 2 — ANALYTICS PIPELINE COMPLETED")
print("=" * 60)
