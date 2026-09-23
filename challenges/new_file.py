import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset
data = {
    'Transaction_Amount': [15, 1200, 35, 2500, 50, 1800, 20, 3000, 45, 2200],
    'Distance_From_Home': [2, 150, 5, 300, 1, 200, 4, 250, 3, 180],
    'Is_Fraud': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# 2. Separate X and y
X = df[['Transaction_Amount', 'Distance_From_Home']]
y = df['Is_Fraud']

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Feature Scaling (Mahalaga sa SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train SVM Classifier
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Predict & Evaluate
y_pred = model.predict(X_test_scaled)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))