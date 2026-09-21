import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data = {
    'CreditScore': [600, 750, 500, 800, 580, 720, 610, 790, 520, 680, 710, 590],
    'Income': [30000, 85000, 20000, 95000, 25000, 70000, 32000, 90000, 22000, 60000, 65000, 28000],
    'Default': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# 2. Features and Target
X = df[['CreditScore', 'Income']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)

# 7. Print Results
print("=== MODEL COMPARISON RESULTS ===")
print(f"Logistic Regression Accuracy : {acc_log * 100:.2f}%")
print(f"k-NN (k=3) Accuracy          : {acc_knn * 100:.2f}%")