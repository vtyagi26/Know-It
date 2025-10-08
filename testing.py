# testing.py
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# 1. Load saved model components
# -------------------------------
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# -------------------------------
# 2. Load test data
# -------------------------------
test_data = pd.read_csv('Testing.csv')
test_data.fillna(0, inplace=True)

# Drop 'prognosis' if present
if 'prognosis' in test_data.columns:
    test_data = test_data.drop('prognosis', axis=1)

# Feature selection + scaling (same as training)
test_sel = selector.transform(test_data)
test_scaled = scaler.transform(test_sel)

# -------------------------------
# 3. Make predictions
# -------------------------------
predictions = model.predict(test_scaled)
predicted_labels = label_encoder.inverse_transform(predictions)

# -------------------------------
# 4. Save predictions to CSV
# -------------------------------
output = pd.DataFrame({
    'Predicted_Prognosis': predicted_labels
})
output.to_csv('Testing_Predictions.csv', index=False)

print("\n✅ Predictions saved to 'Testing_Predictions.csv'")
print("\nSample predictions:")
print(output.head())
