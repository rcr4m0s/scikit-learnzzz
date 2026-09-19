import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'kWh': [100, 150, 200, 250, 300],
    'Bill_PHP': [1200, 1750, 2300, 2850, 3400]
}

df = pd.DataFrame(data)

X = df[['kWh']]
y = df['Bill_PHP']

model = LinearRegression()
model.fit(X, y)

trained = pd.DataFrame({'kWh': [180, 400]})
predict = model.predict(trained)

for train, pred in zip(trained['kWh'], predict):
    print(f"Predicted Bill for {train} kWh: PHP {pred:.2f}")



print(f"\nModel slope (m): {model.coef_[0]:.2f}")
print(f"Model Intercept (b): {model.intercept_:.2f}")