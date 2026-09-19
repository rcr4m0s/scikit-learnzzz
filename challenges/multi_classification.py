import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = {
    'SepalLength': [5.1, 4.9, 6.0, 6.4, 6.3, 6.7, 5.8, 5.7],
    'PetalLength': [1.4, 1.4, 4.5, 4.5, 6.0, 5.8, 5.1, 1.5],
    'Species': [0, 0, 1, 1, 2, 2, 2, 0]
}
df = pd.DataFrame(data)


X = df[['SepalLength', 'PetalLength']]
y = df['Species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

new_flower = pd.DataFrame({
    'SepalLength': [6.2],
    'PetalLength': [4.8]
})

new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
predicted_species = species_map[prediction[0]]

print(f"Predicted Class: {prediction[0]} ({predicted_species})")