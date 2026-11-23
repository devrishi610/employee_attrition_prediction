"""
Rebuild the Employee Attrition model bundle in the CURRENT VS Code environment.

This intentionally follows the final feature-engineered pipeline in the frozen notebook:
- same four engineered features
- stratified 80/20 split (random_state=42)
- StandardScaler + OneHotEncoder(handle_unknown="ignore")
- SMOTE(random_state=42) on training data only
- final Logistic Regression(max_iter=2000, random_state=42)
- threshold 0.50, as used by the notebook's saved model bundle

It does NOT modify the frozen notebook or any existing model file.
"""
from pathlib import Path
import shutil
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "employee_attrition_model_bundle_vscode.joblib"

# Find the dataset without requiring the user to rename it.
candidate_names = [
    "Palo Alto Networks.csv",
    "employee_data.csv",
    "Employee_Data.csv",
    "employee_attrition.csv",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv",
]
csv_path = next((ROOT / name for name in candidate_names if (ROOT / name).exists()), None)

if csv_path is None:
    csv_files = [p for p in ROOT.glob("*.csv")]
    if len(csv_files) == 1:
        csv_path = csv_files[0]
    elif len(csv_files) > 1:
        print("Multiple CSV files found. Please keep the project dataset named 'Palo Alto Networks.csv' or 'employee_data.csv'.")
        print("Found:", [p.name for p in csv_files])
        sys.exit(1)
    else:
        print("ERROR: Project CSV dataset not found in this folder.")
        print("Place the original employee dataset CSV beside this script.")
        sys.exit(1)

print("=" * 65)
print("EMPLOYEE ATTRITION - LOCAL MODEL REBUILD")
print("=" * 65)
print("Dataset:", csv_path.name)
print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Joblib:", joblib.__version__)

df = pd.read_csv(csv_path)
if "Attrition" not in df.columns:
    raise ValueError("Dataset must contain the target column 'Attrition'.")

# Normalize target labels to the notebook's 0=Stayed, 1=Left convention.
if df["Attrition"].dtype == "object":
    target = df["Attrition"].astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "left": 1, "stayed": 0, "1": 1, "0": 0}
    mapped = target.map(mapping)
    if mapped.isna().any():
        raise ValueError("Unrecognized Attrition labels: " + str(sorted(target[mapped.isna()].unique())))
    df["Attrition"] = mapped.astype(int)
else:
    vals = set(pd.Series(df["Attrition"]).dropna().unique().tolist())
    if vals.issubset({0, 1, 0.0, 1.0}):
        df["Attrition"] = df["Attrition"].astype(int)
    else:
        raise ValueError(f"Attrition must be binary 0/1 or Yes/No. Found: {vals}")

# Apply exactly the four feature-engineering formulas in the frozen notebook.
required = [
    "MonthlyIncome", "TotalWorkingYears", "YearsSinceLastPromotion",
    "JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction",
    "RelationshipSatisfaction", "WorkLifeBalance", "OverTime"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing columns required by notebook feature engineering: {missing}")

df = df.copy()
df["IncomeExperienceRatio"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
df["PromotionDelay"] = (df["YearsSinceLastPromotion"] >= 3).astype(int)
df["EngagementScore"] = df[
    ["JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction",
     "RelationshipSatisfaction", "WorkLifeBalance"]
].mean(axis=1)
df["WorkloadStress"] = (
    (df["OverTime"].astype(str).str.strip().str.lower() == "yes") &
    (df["WorkLifeBalance"] <= 2)
).astype(int)

X = df.drop(columns=["Attrition"]).copy()
y = df["Attrition"].copy()

# Retain all predictors as in the frozen notebook's final pipeline.
# Drop rows with missing model inputs/target rather than silently imputing differently.
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].copy()

if y.nunique() != 2:
    raise ValueError("After cleaning, Attrition must contain both classes 0 and 1.")
if y.value_counts().min() < 2:
    raise ValueError("Not enough minority-class records for stratified split/SMOTE.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

final_model = LogisticRegression(max_iter=2000, random_state=42)
final_model.fit(X_train_balanced, y_train_balanced)

class_1_index = list(final_model.classes_).index(1)
test_prob = final_model.predict_proba(X_test_processed)[:, class_1_index]
test_pred = (test_prob >= 0.50).astype(int)

print("\nFRESH MODEL TEST METRICS (threshold=0.50)")
print(f"Accuracy : {accuracy_score(y_test, test_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0)*100:.2f}%")
print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0)*100:.2f}%")
print(f"F1       : {f1_score(y_test, test_pred, zero_division=0)*100:.2f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.3f}")

bundle = {
    "model": final_model,
    "preprocessor": preprocessor,
    "feature_columns": X_train.columns.tolist(),
    "threshold": 0.50,
    "target_mapping": {0: "Stayed", 1: "Left"},
    "metadata": {
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "dataset_file": csv_path.name,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "threshold": 0.50,
        "pipeline_source": "Frozen notebook final feature-engineered pipeline",
    },
}

joblib.dump(bundle, OUTPUT)

# Reload with this same environment and verify preprocessing + predictions.
reloaded = joblib.load(OUTPUT)
X_test_for_reload = X_test[reloaded["feature_columns"]]
X_test_ready = reloaded["preprocessor"].transform(X_test_for_reload)
reload_prob = reloaded["model"].predict_proba(X_test_ready)[:, list(reloaded["model"].classes_).index(1)]
reload_pred = (reload_prob >= reloaded["threshold"]).astype(int)

assert len(reload_pred) == len(y_test)
assert np.isfinite(reload_prob).all()

print("\n" + "=" * 65)
print("MODEL BUNDLE CREATED AND RELOADED SUCCESSFULLY")
print("=" * 65)
print("Saved file:", OUTPUT.name)
print("Model:", type(reloaded["model"]).__name__)
print("Preprocessor:", type(reloaded["preprocessor"]).__name__)
print("Scikit-learn version saved:", reloaded["metadata"]["scikit_learn_version"])
print("Expected input columns:", len(reloaded["feature_columns"]))
print("Test records verified:", len(reload_pred))
print("Predictions generated:", True)
print("\nUse this new file in app.py / check_project.py:")
print(OUTPUT)
