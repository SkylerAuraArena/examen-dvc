import pandas as pd
from sklearn.model_selection import train_test_split
import os

data = pd.read_csv('data/raw_data/raw.csv')

# On supprime la colonne date qui n'est pas utile pour la prédiction
data = data.drop(columns=['date'])

X = data.iloc[:, :-1] 
y = data.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

output_dir = 'data/processed_data/dataset'
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

print("Les ensembles de données ont été créés et sauvegardés dans data/processed_data/dataset")