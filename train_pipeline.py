import pandas as pd
import sys
import urllib.error
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os

print("=== STEP 1: ROBUST DATA INGESTION ===")

sheet_url = "https://docs.google.com/spreadsheets/d/1ykJtioZB3IrKzYKH5oab1Wf_WnCSQ6e7pLZT8nA1uCY/edit?usp=sharing"
csv_export_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
if "/edit" in csv_export_url:
    csv_export_url = csv_export_url.split("/edit")[0] + "/export?format=csv"

try:
    df = pd.read_csv(csv_export_url)
except urllib.error.URLError as e:
    print(f"❌ CRITICAL ERROR: Could not connect to Google Sheets. Check internet. Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to read CSV. Details: {e}")
    sys.exit(1)

if len(df) < 50:
    print(f"❌ CRITICAL ERROR: Dataset only has {len(df)} rows. Aborting to protect production.")
    sys.exit(1)

print(f"✅ SUCCESS: Downloaded {len(df)} responses.")


print("\n=== STEP 2: SAFE DATA PREPROCESSING ===")

# Capture the original question text before renaming
original_columns = list(df.columns)
q_map = {f"q{i}": col for i, col in enumerate(original_columns)}
q_map['q47'] = "stress_level" # The target

# Rename columns dynamically
new_columns = [f"q{i}" for i in range(len(df.columns))]
new_columns[0] = "timestamp"
new_columns[1] = "email"
new_columns[47] = "stress_level"
df.columns = new_columns

df.drop(columns=["timestamp", "email"], inplace=True)
df.dropna(subset=["stress_level"], inplace=True)

valid_stress_labels = ["Rarely or never", "Occasionally", "Frequently"]
df = df[df['stress_level'].isin(valid_stress_labels)]

df = df.fillna("Not Answered")

# Save a raw copy for extracting UI options later
raw_df = df.copy()

frequency_map = {"Rarely or never": 0, "Occasionally": 1, "Frequently": 2, "Not Answered": -1}
time_map = {"Less than 6 hours": 0, "6-7 hours": 1, "7-8 hours": 2, "More than 8 hours": 3, "Not Answered": -1}
involvement_map = {"Not involved at all": 0, "Not very involved": 1, "Somewhat involved": 2, "Very involved": 3, "Not Answered": -1}
yes_no_map = {"No": 0, "Yes": 1, "Not Answered": -1}

df_mapped = df.copy()

for col in df_mapped.columns:
    if col != "stress_level":
        unique_vals = set(df_mapped[col].unique())
        if unique_vals.issubset(set(frequency_map.keys())): df_mapped[col] = df_mapped[col].map(frequency_map)
        elif unique_vals.issubset(set(time_map.keys())): df_mapped[col] = df_mapped[col].map(time_map)
        elif unique_vals.issubset(set(involvement_map.keys())): df_mapped[col] = df_mapped[col].map(involvement_map)
        elif unique_vals.issubset(set(yes_no_map.keys())): df_mapped[col] = df_mapped[col].map(yes_no_map)

X = df_mapped.drop(columns=["stress_level"])
y = df_mapped["stress_level"]

non_mapped_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
if non_mapped_cols:
    safety_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[non_mapped_cols] = safety_encoder.fit_transform(X[non_mapped_cols])

print("✅ SUCCESS: Data is perfectly cleaned, mapped, and mathematically safe.")

print("\n=== STEP 3: DUAL-TRACK ARCHITECTURE (SHADOW MODEL) ===")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get the current UI features
if os.path.exists('questions_schema.json'):
    with open('questions_schema.json', 'r') as f:
        schema = json.load(f)
    ui_selected_names = list(schema.keys())
else:
    ui_selected_names = ['q6', 'q13', 'q17', 'q19', 'q22', 'q36', 'q40', 'q43', 'q51', 'q55', 'q61', 'q64', 'q68', 'q70', 'q73']

# ---------------------------------------------------------
# TRACK A: THE PRODUCTION MODEL (FROZEN UI)
# ---------------------------------------------------------
print("\n⚙️  TRACK A: Training Frozen Production Model...")
X_train_prod = X_train[ui_selected_names].copy()
X_test_prod = X_test[ui_selected_names].copy()

prod_nom = [col for col in X_train_prod.columns if len(X_train_prod[col].unique()) <= 5]
prod_ord = [col for col in X_train_prod.columns if col not in prod_nom]

preprocessor_prod = ColumnTransformer(
    transformers=[
        ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), prod_nom),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), prod_ord)
    ]
)

X_train_prod_final = preprocessor_prod.fit_transform(X_train_prod)
X_test_prod_final = preprocessor_prod.transform(X_test_prod)

model_prod = LogisticRegression(C=0.1, max_iter=2000, solver='lbfgs')
model_prod.fit(X_train_prod_final, y_train)
acc_prod = accuracy_score(y_test, model_prod.predict(X_test_prod_final))
print(f"-> Production Accuracy: {acc_prod:.2f}")


# ---------------------------------------------------------
# TRACK B: THE SHADOW MODEL (DYNAMIC)
# ---------------------------------------------------------
print("\n🕵️  TRACK B: Training Dynamic Shadow Model...")
feature_selector = SelectKBest(score_func=mutual_info_classif, k=15)
X_train_shadow = feature_selector.fit_transform(X_train, y_train)

shadow_selected_names = X_train.columns[feature_selector.get_support(indices=True)]
X_train_shadow_df = pd.DataFrame(X_train_shadow, columns=shadow_selected_names)
X_test_shadow_df = pd.DataFrame(feature_selector.transform(X_test), columns=shadow_selected_names)

shadow_nom = [col for col in X_train_shadow_df.columns if len(X_train_shadow_df[col].unique()) <= 5]
shadow_ord = [col for col in X_train_shadow_df.columns if col not in shadow_nom]

preprocessor_shadow = ColumnTransformer(
    transformers=[
        ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), shadow_nom),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), shadow_ord)
    ]
)

X_train_shadow_final = preprocessor_shadow.fit_transform(X_train_shadow_df)
X_test_shadow_final = preprocessor_shadow.transform(X_test_shadow_df)

model_shadow = LogisticRegression(C=0.1, max_iter=2000, solver='lbfgs')
model_shadow.fit(X_train_shadow_final, y_train)
acc_shadow = accuracy_score(y_test, model_shadow.predict(X_test_shadow_final))
print(f"-> Shadow Accuracy (Max Theoretical): {acc_shadow:.2f}")


# ---------------------------------------------------------
# STEP 4: MONITORING ALERTS & SELF-UPDATING DEPLOYMENT
# ---------------------------------------------------------
print("\n=== STEP 4: SELF-UPDATING DEPLOYMENT ===")

def generate_schema(selected_q_ids, raw_dataframe, q_mapping):
    schema = {}
    for q_id in selected_q_ids:
        text = q_mapping[q_id]
        options = list(raw_dataframe[q_id].unique())
        if "Not Answered" not in options:
            options.append("Not Answered")
        schema[q_id] = {"text": text, "options": [str(opt) for opt in options]}
    return schema

accuracy_gap = acc_shadow - acc_prod

if accuracy_gap > 0.0:  # If Shadow is mathematically superior
    print(f"🚨 UPGRADE TRIGGERED: The Shadow Model is better (+{accuracy_gap:.2f})!")
    print("Automatically generating Version 2.0 of the Streamlit website...")
    
    # Generate and save the new dynamic schema
    new_schema = generate_schema(shadow_selected_names, raw_df, q_map)
    with open('questions_schema.json', 'w') as f:
        json.dump(new_schema, f, indent=4)
        
    # Overwrite production artifacts with Shadow artifacts
    joblib.dump(preprocessor_shadow, 'hybrid_preprocessor.pkl')
    joblib.dump(model_shadow, 'tuned_stress_model.pkl')
    print("✅ Self-Update Complete! The Streamlit UI will dynamically adapt on next refresh.")
else:
    print(f"✅ The current UI (Production Model) is still optimal (Gap: {accuracy_gap:.2f}). No upgrade needed.")
    
    # Ensure the schema file exists even if no upgrade happened
    if not os.path.exists('questions_schema.json'):
        init_schema = generate_schema(ui_selected_names, raw_df, q_map)
        with open('questions_schema.json', 'w') as f:
            json.dump(init_schema, f, indent=4)
            
    joblib.dump(preprocessor_prod, 'hybrid_preprocessor.pkl')
    joblib.dump(model_prod, 'tuned_stress_model.pkl')
    print("✅ Existing model artifacts safely verified and saved.")
