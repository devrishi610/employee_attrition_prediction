import os
import pandas as pd
import joblib

print("=" * 55)
print("EMPLOYEE ATTRITION PROJECT - FILE CHECK")
print("=" * 55)

# 1. Check required files
required_files = [
    "app.py",
    "employee_data.csv",
    "employee_attrition_model_bundle_vscode.joblib"
]

print("\n1. FILE AVAILABILITY")

for file in required_files:
    if os.path.exists(file):
        print(f"FOUND: {file}")
    else:
        print(f"MISSING: {file}")

# 2. Read employee dataset
print("\n2. DATASET CHECK")

try:
    df = pd.read_csv("employee_data.csv")
    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("Total employees:", len(df))
    print("\nColumn names:")
    print(df.columns.tolist())
except Exception as e:
    print("Dataset error:", e)

# 3. Load trained model bundle
print("\n3. MODEL CHECK")

try:
    bundle = joblib.load("employee_attrition_model_bundle_vscode.joblib")

    print("Model bundle loaded successfully!")
    print("Bundle keys:", list(bundle.keys()))

    print("Model:", type(bundle["model"]).__name__)
    print("Preprocessor:", type(bundle["preprocessor"]).__name__)
    print("Threshold:", bundle["threshold"])

    print("Expected feature count:",
          len(bundle["feature_columns"]))

    print("Expected feature columns:")
    print(bundle["feature_columns"])

except Exception as e:
    print("Model error:", e)

print("\n" + "=" * 55)
print("CHECK COMPLETED")
print("=" * 55)
