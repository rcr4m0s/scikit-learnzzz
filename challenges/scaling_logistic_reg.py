import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = {
    'StudyHours': [2, 8, 3, 9, 1, 7, 5, 10],
    'Attendance': [60, 95, 65, 90, 50, 85, 70, 98],
    'Passed': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['StudyHours', 'Attendance']]
y = df['Passed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

new_student = pd.DataFrame({
    'StudyHours': [4, 8],
    'Attendance': [68, 88]
})

new_student_scaler = scaler.transform(new_student)

predictions = model.predict(new_student_scaler)
probabilities = model.predict_proba(new_student_scaler)

print(f"Student A (4 hrs, 68% attendance): Result = {predictions[0]} (Passed Probability {probabilities[0][1]*100:.2f}%)")
print(f"Student A (8 hrs, 88% attendance): Result = {predictions[1]} (Passed Probability {probabilities[1][1]*100:.2f}%)")