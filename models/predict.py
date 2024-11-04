import pandas as pd
import os
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()

with open('models/trained_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

y_pred = trained_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "Mean Squared Error": mse,
    "R2 Score": r2
}

os.makedirs('metrics', exist_ok=True)
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

predictions_df = pd.DataFrame({
    "True Values": y_test,
    "Predictions": y_pred
})

os.makedirs('data/predictions', exist_ok=True)
predictions_df.to_csv('data/predictions/predictions.csv', index=False)

print("Les prédictions ont été sauvegardées dans data/predictions/predictions.csv")
print("Les métriques d'évaluation ont été sauvegardées dans metrics/scores.json")