import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'Link_Count': [1, 5, 2, 8, 10, 1],
    'Word_Offer': [0, 3, 0, 4, 5, 1],
    'Is_Spam': [0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

X = df[['Link_Count', 'Word_Offer']]
y = df['Is_Spam']

model = LogisticRegression()
model.fit(X, y)

links = pd.DataFrame({'Link_Count':[6], 'Word_Offer': [2]})
predictions = model.predict(links)
probabilities = model.predict_proba(links)

for pred, prob in zip(predictions, probabilities):
    status = "Spam" if pred == 0 else "Not Spam"
    pass_prob = prob[1] * 100
    print(f"Result: {status} (Pass Probability: {pass_prob:.2f})")