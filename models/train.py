import pandas as pd
import os
import pickle
from sklearn.linear_model import Ridge

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()

with open('models/best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

print("Les meilleurs paramètres sont :", best_model.get_params())

best_model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
with open('models/trained_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Le modèle entraîné a été sauvegardé dans models/trained_model.pkl")