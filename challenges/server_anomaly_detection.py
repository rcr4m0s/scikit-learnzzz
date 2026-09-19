import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'CPU': [15, 20, 85, 90, 25, 30, 95, 88, 12, 18, 92, 80],
    'RAM': [30, 40, 90, 85, 35, 45, 95, 92, 28, 38, 89, 84],
    'Latency': [10, 12, 150, 180, 15, 11, 200, 160, 8, 14, 175, 140],
    'Anomaly': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['CPU', 'RAM', 'Latency']]
y = df['Anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

print("--- LOGISTICS REGRESSION")
print("ACCURACY:", accuracy_score(y_test, y_pred_log))
print("CONFUSION MATRIX:", confusion_matrix(y_test, y_pred_log))

print("\n--- k-NN Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

new_server = pd.DataFrame([[82, 88, 145]], columns=['CPU', 'RAM', 'Latency'])
new_server_scaled = scaler.transform(new_server)

new_pred = log_model.predict(new_server_scaled)
new_prob = log_model.predict_proba(new_server_scaled)

print("\n--- New Server Prediction ---")
print("Predicted Class:", new_pred[0])
print("Probabilities [Normal, Anomaly]:", new_prob[0])