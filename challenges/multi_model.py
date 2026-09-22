import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 28, 29, 31, 49],
    'EstimatedSalary': [19000, 20000, 43000, 79000, 58000, 36000, 80000, 100000, 43000, 50000, 70000, 89000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "k-Nearest Neighbors (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Gaussian Naïve Bayes": GaussianNB()
}

print("=== ACCURACY COMPARISON ===")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<26}: {acc * 100:.2f}%")

new_customer = pd.DataFrame({'Age': [35], 'EstimatedSalary': [65000]})
new_customer_scaled = scaler.transform(new_customer)

print("\n=== PREDICTION PROBABILITIES FOR NEW CUSTOMER (Age: 35, Salary: $65,000) ===")
log_reg = models["Logistic Regression"]
probabilities = log_reg.predict_proba(new_customer_scaled)[0]

print(f"Probability of NOT Purchasing (Class 0) : {probabilities[0] * 100:.2f}%")
print(f"Probability of Purchasing (Class 1)     : {probabilities[1] * 100:.2f}%")