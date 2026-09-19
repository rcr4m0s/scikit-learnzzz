import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'Monthly_Fee': [299, 499, 899, 999, 299, 899],
    'Support_Tickets': [0, 1, 4, 5, 0, 3],
    'Churn': [0, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)


X = df[['Monthly_Fee', 'Support_Tickets']]
y = df['Churn']


model = LogisticRegression()
model.fit(X, y)


new_customers = pd.DataFrame({
    'Monthly_Fee': [299, 899],
    'Support_Tickets': [3, 0]
})

predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)


labels = ['Customer A', 'Customer B']
for label, pred, prob in zip(labels, predictions, probabilities):
    status = "Will Churn (Leave)" if pred == 1 else "Will Stay"
    churn_prob = prob[1] * 100
    print(f"{label} -> Result: {status} (Churn Probability: {churn_prob:.2f}%)")