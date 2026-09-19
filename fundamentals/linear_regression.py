import pandas as pd
from sklearn.linear_model import LinearRegression 

data = {
    'Experience_Years': [1, 2, 3, 4, 5],
    'Salary_PHP': [25000, 30000, 38000, 45000, 53000]
}

df = pd.DataFrame(data)

X = df[['Experience_Years']]
y = df['Salary_PHP']

model = LinearRegression()
model.fit(X, y)

new_experience = pd.DataFrame({'Experience_Years': [6, 10]})
predictions = model.predict(new_experience)

for exp, pred in zip(new_experience['Experience_Years'], predictions):
    print(f"Predicted Salary for {exp} years: PHP {pred:.2f}")


print(f"\nModel Slope (m): {model.coef_[0]:.2f}")
print(f"Model Intercept (b): {model.intercept_:.2f}")