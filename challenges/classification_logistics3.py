from sklearn.linear_model import LogisticRegression


X = [
    [0.15, 2],  # Motorcycle
    [0.20, 2],  # Motorcycle
    [1.2,  4],  # Sedan
    [1.5,  4],  # Sedan
    [2.5,  4],  # SUV
    [3.0,  4]   # SUV
]

y = [0, 0, 1, 1, 2, 2]

model = LogisticRegression()
model.fit(X, y)


new_vehicles = [
    [0.18, 2],
    [2.8, 4]
]
predictions = model.predict(new_vehicles)

labels_map = {0: "Motorcycle", 1: "Sedan", 2: "SUV"}

for vehicle, pred in zip(new_vehicles, predictions):
    vehicle_type = labels_map[pred]
    print(f"Sasakyan {vehicle} -> Hula: {vehicle_type}")