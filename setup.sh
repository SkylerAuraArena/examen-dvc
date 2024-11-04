#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Redirection de toutes les sorties vers setup.log en plus de l'affichage dans la sortie standard
exec > >(tee -a setup.log) 2>&1

# Variables
DATA_DIR="$(pwd)/data"        # Utilisation d'un chemin absolu pour éviter les ambiguïtés
MODELS_DIR="$(pwd)/models"

python $DATA_DIR/preprocessing.py

python $DATA_DIR/scalling.py

python $MODELS_DIR/grid_search.py

python $MODELS_DIR/train.py

python $MODELS_DIR/predict.py

echo "Traitement terminé avec succès."

#______________________________________________________________________________________________________________________________
