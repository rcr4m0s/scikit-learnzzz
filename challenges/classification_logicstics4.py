import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = {
    'Age': [22, 25, 47, 52, 46, 56, 21, 32],
    'EstimatedSalary': [20000, 35000, 80000, 150000, 62000, 90000, 18000, 60000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = LogisticRegression()
model.fit(X_scaled, y)

new_customer = pd.DataFrame({
    'Age': [30, 50], 
    'EstimatedSalary': [50000, 100000]
})

new_customer_scaled = scaler.transform(new_customer)

predictions = model.predict(new_customer_scaled)
probabilities = model.predict_proba(new_customer_scaled)

print(f"Customer A (Age 30, Salary 50k): Purchased = {predictions[0]} (Probability: {probabilities[0][1]*100:.2f}%)")
print(f"Customer B (Age 50, Salary 100k): Purchased = {predictions[1]} (Probability: {probabilities[1][1]*100:.2f}%)")