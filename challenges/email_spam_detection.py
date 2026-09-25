import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Dataset
data = {
    'Link_Count': [1, 12, 2, 15, 0, 8, 3, 20, 1, 10],
    'Capital_Word_Count': [2, 45, 1, 60, 0, 30, 4, 50, 3, 35],
    'Is_Spam': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# 2. Separate X and y
X = df[['Link_Count', 'Capital_Word_Count']]
y = df['Is_Spam']

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Train Naïve Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Predict & Evaluate
y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))