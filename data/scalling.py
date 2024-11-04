import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

scaler = StandardScaler()

# Sélectionner uniquement les colonnes numériques
X_train_scaled = X_train.select_dtypes(include=['number'])
X_test_scaled = X_test.select_dtypes(include=['number'])

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)

print("Les ensembles normalisés ont été sauvegardés dans data/processed_data")