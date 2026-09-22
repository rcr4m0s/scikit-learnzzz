import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data = {
    'Age': [23, 45, 18, 52, 30, 40, 22, 60, 48, 28, 35, 50],
    'DebtToIncome': [0.45, 0.12, 0.80, 0.05, 0.35, 0.20, 0.65, 0.10, 0.15, 0.50, 0.25, 0.08],
    'HighRisk': [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]  # 1 = High Risk, 0 = Low Risk
}
df = pd.DataFrame(data)

X = df[['Age', 'DebtToIncome']]
y = df['HighRisk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

print("=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))

print("\n=== k-NEAREST NEIGHBORS (k=3) ===")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))