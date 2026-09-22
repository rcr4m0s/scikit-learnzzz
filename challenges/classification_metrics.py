import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

data = {
    'WordCount': [120, 50, 800, 450, 30, 900, 150, 600, 40, 700],
    'LinkCount': [1, 0, 8, 5, 0, 10, 2, 7, 0, 9],
    'IsSpam': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)


X = df[['WordCount', 'LinkCount']]
y = df['IsSpam']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y, y_pred, target_names=['Not Spam', 'Spam']))