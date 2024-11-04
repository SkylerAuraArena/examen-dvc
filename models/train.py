import pandas as pd
import os
import pickle
from sklearn.linear_model import Ridge

X_train = pd.read_csv('data/processed_data/scalled_dataset/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/dataset/y_train.csv').values.ravel()

with open('models/best_params/best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

print("Les meilleurs paramètres sont :", best_model.get_params())

best_model.fit(X_train, y_train)

output_dir = 'models/trained_model'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/trained_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Le modèle entraîné a été sauvegardé dans models/trained_model/trained_model.pkl")