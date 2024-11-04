import pandas as pd
from sklearn.model_selection import train_test_split
import os

data = pd.read_csv('data/raw_data/raw.csv')

X = data.iloc[:, :-1] 
y = data.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

print("Les ensembles de données ont été créés et sauvegardés dans data/processed_data")