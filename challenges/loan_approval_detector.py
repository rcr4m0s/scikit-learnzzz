import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = {
    'CreditScore': [600, 750, 580, 800, 620, 700, 550, 780],
    'Income': [300000, 800000, 250000, 1200000, 400000, 650000, 200000, 950000],
    'Approved': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['CreditScore', 'Income']]
y = df['Approved']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

new_applicants = pd.DataFrame({
    'CreditScore': [610, 720],
    'Income': [350000, 750000]
})


new_applicants_scaler = scaler.transform(new_applicants)

predictions = model.predict(new_applicants_scaler)
probabilities = model.predict_proba(new_applicants_scaler)

print(f"Applicant 1: Credit Score = {predictions[0]}, Probability = {(probabilities[0][1] * 100):.2f}%")
print(f"Applicant 2: Credit Score = {predictions[1]}, Probability = {(probabilities[1][1] * 100):.2f}%")

