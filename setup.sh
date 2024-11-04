#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Rediriger toutes les sorties vers setup.log et afficher à l'écran
exec > >(tee -a setup.log) 2>&1

# Variables
FEATURES_DIR="$(pwd)/src/features"        # Utiliser un chemin absolu pour éviter les ambiguïtés

python $FEATURES_DIR/preprocessing.py

python $FEATURES_DIR/scalling.py

echo "Traitement terminé avec succès."

#______________________________________________________________________________________________________________________________
