import mlflow
import mlflow.sklearn
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import re

# -------------------------------------------------
# Configuration MLflow & Chargement du modèle et du seuil
# -------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "Best Model"
MODEL_VERSION = 2  # À ajuster selon la version souhaitée

# Chargement du modèle depuis le Model Registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
best_model = mlflow.sklearn.load_model(model_uri)
print(f"✅ Modèle '{MODEL_NAME}' v{MODEL_VERSION} chargé depuis MLflow.")

# Chargement du seuil optimal depuis le fichier pickle
THRESHOLD_PATH = "best_threshold.pkl"
with open(THRESHOLD_PATH, "rb") as f:
    best_threshold = pickle.load(f)
print(f"✅ Seuil optimal chargé : {best_threshold}")

# -------------------------------------------------
# Liste fixe des 18 colonnes attendues par le modèle
# (le modèle a été entraîné sur ces colonnes, TARGET n'est pas inclus)
# -------------------------------------------------
selected_features = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CNT_FAM_MEMBERS", 
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", 
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", 
    "INCOME_PER_PERSON", "PAYMENT_RATE", "DAYS_EMPLOYED_RATIO", "AGE_YEARS", 
    "EMPLOYED_YEARS", "YEARS_ID_PUBLISH", "YEARS_REGISTRATION"
]

# -------------------------------------------------
# Fonction pour nettoyer les noms de colonnes
# -------------------------------------------------
def clean_column_names(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df

# -------------------------------------------------
# Fonction de prétraitement des features
# -------------------------------------------------
def preprocess_features(df):
    """
    Applique le prétraitement suivant :
      - Pour DAYS_EMPLOYED : remplace 365243 par NaN.
      - Impute les autres colonnes numériques avec la médiane.
      - Encode les variables catégoriques via get_dummies (drop_first=True).
      - Crée les nouvelles features attendues.
      - Vérifie que toutes les colonnes définies dans selected_features sont présentes.
      - Réorganise les colonnes dans l'ordre défini et normalise via le scaler sauvegardé.
    """
    df = df.copy()
    
    # Nettoyage des noms de colonnes
    df = clean_column_names(df)
    
    # Traitement spécifique pour DAYS_EMPLOYED : remplacement de 365243 par NaN
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    
    # Imputation des autres colonnes numériques (excluant DAYS_EMPLOYED)
    num_cols = [col for col in df.select_dtypes(include=["number"]).columns if col != "DAYS_EMPLOYED"]
    imputer = SimpleImputer(strategy="median")
    if num_cols:
        df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Encodage des variables catégoriques (si présentes)
    df = pd.get_dummies(df, drop_first=True)
    
    # Création des nouvelles features
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"] / -365
    if "DAYS_ID_PUBLISH" in df.columns:
        df["YEARS_ID_PUBLISH"] = df["DAYS_ID_PUBLISH"] / -365
    if "DAYS_REGISTRATION" in df.columns:
        df["YEARS_REGISTRATION"] = df["DAYS_REGISTRATION"] / -365

    # Vérification que toutes les colonnes attendues sont présentes
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        raise ValueError("Les colonnes suivantes sont manquantes dans les données : " + ", ".join(missing_cols))
    
    # Réorganisation des colonnes dans l'ordre défini
    df = df[selected_features]
    
    # Chargement du scaler sauvegardé et transformation
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    df[selected_features] = scaler.transform(df[selected_features])
    
    return df

# -------------------------------------------------
# Création de l'API Flask
# -------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API MLflow active 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Le JSON doit contenir une clé 'features'."}), 400
        
        # Conversion des données entrantes en DataFrame
        features = pd.DataFrame(data["features"])
        
        # Prétraitement
        features_processed = preprocess_features(features)
        
        # Prédiction de probabilité (on suppose que predict_proba retourne un tableau de forme (n_samples, 2))
        y_pred_prob = best_model.predict_proba(features_processed)[:, 1]
        
        # Application du seuil optimal : crédit accordé si la probabilité est inférieure au seuil
        y_pred = (y_pred_prob < best_threshold).astype(int)
        decisions = ["Crédit accordé" if pred == 1 else "Crédit refusé" for pred in y_pred]
        
        response = {
            "probability": y_pred_prob.tolist(),
            "prediction": y_pred.tolist(),
            "decision": decisions
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # On utilise le port 5002 pour éviter le conflit avec l'UI MLflow sur le port 5001
    app.run(host="0.0.0.0", port=5002, debug=True)
