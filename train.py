# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- File paths ---
DATA_PATH = r"E:\Code alpha Internship\Car price prediction\car data.csv"
MODEL_PATH = r"E:\Code alpha Internship\Car price prediction\car_price_pipeline.pkl"

# --- 1. Load data ---
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# --- 2. Prepare features/
X = df.drop(columns=["Car_Name", "Selling_Price"])
y = df["Selling_Price"]

# Define feature groups
numeric_features = ["Year", "Present_Price", "Driven_kms", "Owner"]
categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]

# --- 3. Preprocessing + pipeline ---
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)  # ✅ FIXED
], remainder="drop")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# --- 4. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Fit ---
model.fit(X_train, y_train)

# --- 6. Evaluate on test set ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation (test set):")
print(f" MAE:  {mae:.4f}")
print(f" MSE:  {mse:.4f}")
print(f" RMSE: {rmse:.4f}")
print(f" R2:   {r2:.4f}")

# --- 7. Cross-validation (optional) ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"5-fold CV R2 scores: {cv_scores}")
print(f"CV R2 mean: {cv_scores.mean():.4f}")

# --- 8. Feature importance ---
ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + ohe_feature_names
importances = model.named_steps["regressor"].feature_importances_

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop features by importance:")
print(imp_df.head(10).to_string(index=False))

# --- 9. Save pipeline ---
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Pipeline saved to: {os.path.abspath(MODEL_PATH)}")
