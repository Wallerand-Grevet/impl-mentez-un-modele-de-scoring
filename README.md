# API de Scoring Crédit

Cette API permet de prédire le risque de défaut de crédit à partir de données clients. Elle utilise un modèle de machine learning préalablement entraîné (LightGBM) et intégré dans le projet, avec un seuil optimal défini pour la prise de décision.

## Objectif du projet

- Offrir une interface simple (via une requête HTTP POST) pour obtenir un score de crédit.
- Intégrer un prétraitement des données conforme à celui utilisé pendant l'entraînement.
- Utiliser un modèle embarqué pour éviter les dépendances cloud (MLflow en option).
- Fournir un système de prédiction facilement intégrable dans un dashboard ou une application cliente.

## Structure du projet

.
├── .github/workflows/             # GitHub Actions pour tests
├── .gitignore                     # Fichiers à ignorer par Git
├── app.py                         # Script principal Flask
├── best_model.pkl                 # Modèle LightGBM entraîné
├── best_threshold.pkl             # Seuil optimal
├── model_ok.ipynb                 # Notebook test modèle
├── requirements.txt               # Dépendances Python
├── scaler.pkl                     # Scaler (MinMaxScaler)
├── test_api.py                    # Tests unitaires
└── README.md                      # Présentation du projet (à compléter)
