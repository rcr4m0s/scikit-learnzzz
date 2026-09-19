import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


data = {
    'Age': [25, 30, 45, 50, 65, 70, 40, 60],
    'Cholesterol': [180, 190, 220, 240, 280, 300, 210, 260],
    'RiskLevel': [0, 0, 1, 1, 2, 2, 0, 2]
}
df = pd.DataFrame(data)

X = df[['Age', "Cholesterol"]]
y = df['RiskLevel']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

new_patient = pd.DataFrame({
    'Age': [55],
    'Cholesterol': [250]
})

new_patient_scaler = scaler.transform(new_patient)
predictions = model.predict(new_patient_scaler)

map_risk = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
predicted_risk = map_risk[predictions[0]]

print(f"Predicted Risk: {predictions[0]} ({predicted_risk})")