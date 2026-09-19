import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'Study_Hours': [1, 2, 3, 5, 6, 8, 9],
    'Passed': [0, 0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['Study_Hours']]
y = df['Passed']

model = LogisticRegression()
model.fit(X, y)

new_students = pd.DataFrame({'Study_Hours': [2.5, 7]})
predictions = model.predict(new_students)
probabilities = model.predict_proba(new_students)

for hours, pred, prob in zip(new_students['Study_Hours'], predictions, probabilities):
    status = 'Pass' if pred == 1 else 'Fail'
    pass_prob = prob[1] * 100
    print(f"Study Hours: {hours} -> Result: {status} (Pass Probability: {pass_prob:.2f})%")