import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

data = {
    'CPU_Usage': [15, 20, 35, 40, 65, 70, 85, 90, 25, 60, 80, 95],
    'Temperature': [40, 45, 50, 55, 68, 72, 85, 90, 48, 70, 82, 92],
    'Status': [0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 2, 2]
}

df = pd.DataFrame(data)

X = df[['CPU_Usage', 'Temperature']]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:", classification_report(y_test, y_pred))
