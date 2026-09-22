import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Monthly_Bill': [20, 100, 25, 110, 80, 30, 95, 105, 35, 85, 90, 40],
    'Tenure_Months': [36, 2, 48, 1, 12, 24, 3, 5, 60, 8, 4, 18],
    'Churn': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Monthly_Bill', 'Tenure_Months']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_log = LogisticRegression()
model_log.fit(X_train_scaled, y_train)
model_log_pred = model_log.predict(X_test_scaled)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)

print("=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, model_log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, model_log_pred))

print("\n=== k-NEAREST NEIGHBORS (k=3) ===")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))