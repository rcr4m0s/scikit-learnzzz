import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Packet_Size': [100, 1200, 1500, 850, 200, 1100, 1300, 150, 900, 1400],
    'Connection_Duration': [0.5, 12.0, 15.5, 8.0, 0.2, 10.5, 14.0, 0.4, 9.2, 16.0],
    'Failed_Logins': [0, 3, 5, 2, 0, 4, 5, 0, 2, 4],
    'Is_Attack': [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[['Packet_Size', 'Connection_Duration', 'Failed_Logins']]
y = df['Is_Attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:", confusion_matrix(y_test, y_predict))