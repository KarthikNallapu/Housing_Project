import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from joblib import dump
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv('Hyderabad.csv')
df.columns = df.columns.str.strip()

# Define target and features
target = 'Price'
categorical_features = ['Location']
numeric_features = ['Area', 'No. of Bedrooms', 'Resale']
amenities = [col for col in df.columns if col not in [target] + categorical_features + numeric_features]

# Optional: Drop rows with too many missing values
df.dropna(thresh=len(df.columns)-5, inplace=True)

# Add new feature: total number of amenities
df['TotalAmenities'] = df[amenities].sum(axis=1)
numeric_features.append('TotalAmenities')

# Prepare data
X = df.drop(target, axis=1)
y = df[target]

# Preprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('amen', 'passthrough', amenities)
])

# Pipeline with regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):,.2f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")

# Save model
with open('hyderabad_housing_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved.")

# Feature importance (Optional but useful)
model = pipeline.named_steps['regressor']
try:
    feature_names = (
        numeric_features +
        list(pipeline.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_features)) +
        amenities
    )
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-15:]  # Top 15

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Top Features Influencing House Price")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("📊 Feature importance plot saved.")
except Exception as e:
    print(f"⚠️ Could not compute feature importance: {e}")
