# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import joblib

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
data = pd.read_csv('Testing.csv')  # using your available dataset
data.fillna(0, inplace=True)

label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])
print("Classes (Prognosis):", list(label_encoder.classes_))

X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Balance dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Feature selection
selector = SelectKBest(f_classif, k=min(30, X_train.shape[1]))  # pick top 30 or less
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

# -------------------------------
# 2. Model definition
# -------------------------------
model = XGBClassifier(
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

param_dist = {
    'n_estimators': randint(200, 800),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 15),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 1)
}

# -------------------------------
# 3. Hyperparameter tuning
# -------------------------------
print("\n🔍 Running RandomizedSearchCV for hyperparameter optimization...")
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
random_search.fit(X_train_scaled, y_train)
best_params = random_search.best_params_
print("\n✅ Best Parameters:", best_params)

# -------------------------------
# 4. Train with best model
# -------------------------------
best_model = XGBClassifier(**best_params, eval_metric='mlogloss', use_label_encoder=False)
best_model.fit(X_train_scaled, y_train)

y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
print(f"\n📊 Cross-Validation Mean Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"📉 Std Deviation: {cv_scores.std() * 100:.2f}%")

# -------------------------------
# 5. Save model artifacts
# -------------------------------
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\n✅ Model, scaler, selector, and label encoder saved successfully.")
