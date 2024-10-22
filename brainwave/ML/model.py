from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('Testing.csv')

data.fillna(0, inplace=True)

label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])
print("Classes (Prognosis):", label_encoder.classes_)

X = data.drop('prognosis', axis=1)
y = data['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=62)

model = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=62)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters from Grid Search:", best_params)

# Train with best parameters
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Accuracy with best parameters: ", 100 * accuracy_best)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", 100 * accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# New patient prediction (example)
new_patient = [[0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0]]
predicted_illness = model.predict(new_patient)
predicted_label = label_encoder.inverse_transform(predicted_illness)
print(f'Predicted illness for new patient: {predicted_label[0]}')

# Cross-validation to get a more stable accuracy
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Accuracy: ", 100 * cv_scores.mean())



# Check for class balance
print(data['prognosis'].value_counts())

# If imbalanced, use SMOTE for oversampling the minority class
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=62)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.15, random_state=62)

# Hyperparameter tuning with a broader range
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)







