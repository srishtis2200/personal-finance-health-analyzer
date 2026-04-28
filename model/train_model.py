import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

#load processed dataset
df = pd.read_csv('data/processed/finance_processed.csv')

print("Shape:", df.shape)
print("\nLabel distribution:")
print(df['health_label'].value_counts())

#prepare x and y
X = df.drop('health_label', axis=1)
y = df['health_label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nClasses:", le.classes_)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

#feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nScaling done!")

#model training and comparision
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

#saving best suited model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n{'='*40}")
print(f"Best Model: {best_model_name}")
print(f"Best F1 Score: {results[best_model_name]:.4f}")

joblib.dump(best_model, 'model/finance_model.pkl')
joblib.dump(scaler,     'model/scaler.pkl')
joblib.dump(le,         'model/label_encoder.pkl')

print("\nSaved:")
print("  model/finance_model.pkl")
print("  model/scaler.pkl")
print("  model/label_encoder.pkl")