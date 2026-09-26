import pandas as pd
from sklearn.naive_bayes import GaussianNB


df = pd.DataFrame({'Time_On_Site': [2, 15, 3, 20, 5, 25], 'Purchased': [0, 1, 0, 1, 0, 1]})

model = GaussianNB().fit(df[['Time_On_Site']], df['Purchased'])

new_user = pd.DataFrame({'Time_On_Site': [18]})
pred = model.predict(new_user)[0]
prob = model.predict_proba(new_user)[0]

print(f"Prediction: {pred} | Probabilities [Class 0, Class 1]: {prob}")