import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


data = {
    'Temperature': [65, 85, 90, 55, 95, 70, 88, 60, 100, 75, 82, 58],
    'Vibration':   [1.2, 3.5, 4.0, 0.8, 4.5, 1.5, 3.8, 1.0, 5.0, 2.0, 3.2, 0.9],
    'Failed':      [0,   1,   1,   0,   1,   0,   1,   0,   1,   0,   1,   0]
}
df = pd.DataFrame(data)

X = df[['Temperature', 'Vibration']]
y = df['Failed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))