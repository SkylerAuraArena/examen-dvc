import pandas as pd
import os
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

X_train = pd.read_csv('data/processed_data/scalled_dataset/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/dataset/y_train.csv').values.ravel()

# Créer un pipeline avec un imputer et le modèle Ridge
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Remplace les NaN par la moyenne
    ('ridge', Ridge())
])

# Grille de paramètres pour Ridge
param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0],
    'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

output_dir = 'models/best_params'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Les meilleurs paramètres sont :", best_params)
print("Le modèle optimisé a été sauvegardé dans models/best_model.pkl")