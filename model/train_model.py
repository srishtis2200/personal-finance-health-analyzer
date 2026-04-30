import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

#load dataset
df = pd.read_csv('data/processed/finance_processed_raw.csv')

print("Shape:", df.shape)
print("\nLabel distribution:")
print(df['health_label'].value_counts())

#prepare x and y
x = df.drop('health_label', axis=1)
y = df['health_label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nClasses:", le.classes_)

#train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(x_train)}")
print(f"Testing samples:  {len(x_test)}")

#model training
# LR needs scaling → Pipeline handles it internally
# RF and XGBoost are tree models → no scaling needed
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ]),

    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42
    )
}

#cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results    = {}
cv_results = {}

#training and evaluation
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")

    # CV on raw data — Pipeline handles scaling internally for LR
    # RF and XGBoost don't need scaling
    cv_scores = cross_val_score(
        model, x_train, y_train,
        cv=cv,
        scoring='f1_weighted'
    )

    print(f"CV F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training data
    if name == 'XGBoost':
        # Correct multiclass imbalance handling for XGBoost
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(x_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(x_train, y_train)

    #predict on test data
    y_pred = model.predict(x_test)

    test_f1 = f1_score(y_test, y_pred, average='weighted')
    results[name]    = test_f1
    cv_results[name] = cv_scores.mean()

    print(f"Test F1: {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    #feature importance for tree models
    if name == 'Random Forest':
        importance_df = pd.DataFrame({
            'feature':    x.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("\nTop 5 Important Features:")
        print(importance_df.head())

    elif name == 'XGBoost':
        importance_df = pd.DataFrame({
            'feature':    x.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("\nTop 5 Important Features:")
        print(importance_df.head())

#select best model based on CV
best_model_name = max(cv_results, key=cv_results.get)
best_model      = models[best_model_name]

print(f"\n{'='*60}")
print(f"✅ Best Model (CV): {best_model_name}")
print(f"✅ CV F1:           {cv_results[best_model_name]:.4f}")
print(f"✅ Test F1:         {results[best_model_name]:.4f}")

#retrain best model on full data
print("\nRetraining best model on full dataset...")
if best_model_name == 'XGBoost':
    sample_weights = compute_sample_weight('balanced', y_encoded)
    best_model.fit(x, y_encoded, sample_weight=sample_weights)
else:
    best_model.fit(x, y_encoded)

#save artefacts
os.makedirs('model', exist_ok=True)

joblib.dump(best_model, 'model/finance_model.pkl')
joblib.dump(le,         'model/label_encoder.pkl')

print("\nSaved:")
print("  ✅ model/finance_model.pkl")
print("  ✅ model/label_encoder.pkl")