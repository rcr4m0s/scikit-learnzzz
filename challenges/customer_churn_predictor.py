import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


data = {

    'MonthlyFee': [29.99, 89.99, 49.99, 119.99, 19.99, 99.99, 39.99, 109.99],
    'TenureMonths': [36, 2, 24, 1, 48, 5, 18, 3],
    'Churn': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['MonthlyFee', 'TenureMonths']]
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

new_customers = pd.DataFrame({
    'MonthlyFee': [35.00, 95.00],
    'TenureMonths': [28, 4]
})

new_customers_scaler = scaler.transform(new_customers)

predictions = model.predict(new_customers_scaler)
probabilities = model.predict_proba(new_customers_scaler)

print(f"Customer A: Churn = {predictions[0]} Probability = {(probabilities[0][1] * 100):.2f}%")
print(f"Customer B: Churn = {predictions[1]} Probability = {(probabilities[1][1] * 100):.2f}%")