from sklearn.linear_model import LogisticRegression

X = [
    [50, 1],
    [60, 2],
    [80, 2],
    [200, 4],
    [250, 5],
    [300, 5]
]

y = [0, 0, 0, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)

new_house = [[220, 4]]
predictions = model.predict(new_house)

status = "Luxury" if predictions[0] == 1 else "Budget"

print(f"Guess: {status}")