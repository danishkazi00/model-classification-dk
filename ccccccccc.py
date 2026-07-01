# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import LabelEncoder

# ==========================================
# SET STYLE
# ==========================================

sns.set_style("whitegrid")

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading Dataset...")

df = pd.read_csv(
    r"C:\Users\dell\PycharmProjects\PythonProject\llm_model_capability_classification.csv"
)

print("Dataset Loaded Successfully!")

# ==========================================
# FIRST 5 ROWS
# ==========================================

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# COLUMN NAMES
# ==========================================

print("\nColumns:")
print(df.columns)

# ==========================================
# DATASET INFO
# ==========================================

print("\nDataset Info:")
print(df.info())

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

df = df.fillna("Unknown")

# ==========================================
# TARGET COLUMN
# ==========================================

# Change target column if needed
target_col = "supports_reasoning"

# ==========================================
# CONVERT TARGET TO NUMERIC
# ==========================================

le_target = LabelEncoder()

df[target_col] = le_target.fit_transform(
    df[target_col].astype(str)
)

print("\nTarget Unique Values:")
print(df[target_col].unique())

# ==========================================
# DROP UNNECESSARY COLUMNS
# ==========================================

drop_cols = []

# Drop if column exists
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# ==========================================
# TARGET DISTRIBUTION
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(x=df[target_col])

plt.title("Target Distribution")

plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ==========================================
# SIMPLE IV CALCULATION FUNCTION
# ==========================================

def calc_iv(dataframe, feature, target):

    temp_df = dataframe[[feature, target]].copy()

    # Numeric column binning
    if (
        pd.api.types.is_numeric_dtype(temp_df[feature])
        and temp_df[feature].nunique() > 10
    ):

        temp_df["bin"] = pd.qcut(
            temp_df[feature],
            q=10,
            duplicates="drop"
        )

    else:
        temp_df["bin"] = temp_df[feature].astype(str)

    grouped = temp_df.groupby("bin")[target].agg(
        ["count", "sum"]
    )

    grouped.columns = ["total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["good"] = grouped["good"].replace(0, 0.5)
    grouped["bad"] = grouped["bad"].replace(0, 0.5)

    grouped["pct_good"] = (
        grouped["good"] / grouped["good"].sum()
    )

    grouped["pct_bad"] = (
        grouped["bad"] / grouped["bad"].sum()
    )

    grouped["woe"] = np.log(
        grouped["pct_good"] / grouped["pct_bad"]
    )

    grouped["iv"] = (
        grouped["pct_good"] - grouped["pct_bad"]
    ) * grouped["woe"]

    return grouped["iv"].sum()

# ==========================================
# CALCULATE IV VALUES
# ==========================================

print("\nCalculating IV Values...")

iv_dict = {}

features = [
    col for col in df.columns
    if col != target_col
]

for col in features:

    try:
        iv_value = calc_iv(
            df,
            col,
            target_col
        )

        iv_dict[col] = iv_value

    except:
        pass

iv_df = pd.DataFrame(
    list(iv_dict.items()),
    columns=["Feature", "IV"]
)

iv_df = iv_df.sort_values(
    by="IV",
    ascending=False
)

print("\nInformation Value Table:")
print(iv_df)

# ==========================================
# SELECT FEATURES USING IV
# ==========================================

selected_features = iv_df[
    iv_df["IV"] > 0.02
]["Feature"].tolist()

print("\nSelected Features:")
print(selected_features)

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df[selected_features]

# Convert all categorical/text columns
X = pd.get_dummies(
    X,
    drop_first=True
)

y = df[target_col]

print("\nProcessed Features:")
print(X.columns)

print("\nFeature Shape:")
print(X.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Test Split Completed")

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(X_train, y_train)

# Prediction
lr_pred = lr.predict(X_test)

# ==========================================
# LOGISTIC REGRESSION RESULTS
# ==========================================

print("\n===== Logistic Regression =====")

print(
    "Accuracy:",
    accuracy_score(y_test, lr_pred)
)

print(
    "Precision:",
    precision_score(y_test, lr_pred)
)

print(
    "Recall:",
    recall_score(y_test, lr_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, lr_pred)
)

# ==========================================
# DECISION TREE
# ==========================================

print("\nTraining Decision Tree...")

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

# Prediction
dt_pred = dt.predict(X_test)

# ==========================================
# DECISION TREE RESULTS
# ==========================================

print("\n===== Decision Tree =====")

print(
    "Accuracy:",
    accuracy_score(y_test, dt_pred)
)

print(
    "Precision:",
    precision_score(y_test, dt_pred)
)

print(
    "Recall:",
    recall_score(y_test, dt_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, dt_pred)
)

# ==========================================
# GRID SEARCH CV
# ==========================================

print("\nRunning GridSearchCV...")

param_grid = {

    "max_depth": [3, 5, 7, 10],

    "min_samples_split": [2, 5, 10],

    "criterion": ["gini", "entropy"]

}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

# ==========================================
# BEST PARAMETERS
# ==========================================

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Score:")
print(grid.best_score_)

# ==========================================
# BEST MODEL
# ==========================================

best_model = grid.best_estimator_

best_pred = best_model.predict(X_test)

# ==========================================
# FINAL MODEL EVALUATION
# ==========================================

print("\n===== Final Tuned Decision Tree =====")

print(
    "Accuracy:",
    accuracy_score(y_test, best_pred)
)

print(
    "Precision:",
    precision_score(y_test, best_pred)
)

print(
    "Recall:",
    recall_score(y_test, best_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, best_pred)
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(y_test, best_pred)
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_pred
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": best_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

# ==========================================
# FEATURE IMPORTANCE GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Feature Importance")

plt.show()

# ==========================================
# FINAL CONCLUSION
# ==========================================

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")

print("Dataset Loaded Successfully")
print("IV Calculation Completed")
print("Logistic Regression Completed")
print("Decision Tree Completed")
print("GridSearchCV Completed")
print("Accuracy and F1 Score Calculated")
