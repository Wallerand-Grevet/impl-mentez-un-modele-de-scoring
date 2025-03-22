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
MLFLOW_TRACKING_URI = "https://mlflow-server-hzm1.onrender.com"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "Best Model"
MODEL_VERSION = 1  # À ajuster selon la version souhaitée

# Chargement du modèle depuis le Model Registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
best_model = mlflow.sklearn.load_model(model_uri)
print(f"✅ Modèle '{MODEL_NAME}' v{MODEL_VERSION} chargé depuis MLflow.")
print(f"✅ Type du modèle chargé : {type(best_model)}")


# Chargement du seuil optimal depuis MLflow
client = MlflowClient()

# Récupère la dernière version validée du modèle (ou la version 1 ici)
model_version_details = client.get_model_version(name=MODEL_NAME, version=str(MODEL_VERSION))
run_id = model_version_details.run_id

# Récupère les paramètres du run
run = client.get_run(run_id)
best_threshold = float(run.data.params["Best Threshold"])

print(f"✅ Seuil optimal récupéré depuis MLflow : {best_threshold}")

# -------------------------------------------------
# Liste fixe des 18 colonnes attendues par le modèle
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
    df = df.copy()
    
    # Nettoyage des noms de colonnes
    df = clean_column_names(df)
    
    # Traitement spécifique pour DAYS_EMPLOYED : remplacement de 365243 par NaN
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    
    # Imputation des autres colonnes numériques
    num_cols = [col for col in df.select_dtypes(include=["number"]).columns if col != "DAYS_EMPLOYED"]
    imputer = SimpleImputer(strategy="median")
    if num_cols:
        df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Encodage des variables catégoriques
    df = pd.get_dummies(df, drop_first=True)
    
    # ✅ **Ajout des colonnes manquantes avec valeurs NaN si absentes**
    for col in selected_features:
        if col not in df.columns:
            df[col] = np.nan
    
    # ✅ **Création des nouvelles features**
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

    # 🔍 **Debug : Voir quelles colonnes sont manquantes avant normalisation**
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        print(f"❌ Colonnes manquantes AVANT normalisation : {missing_cols}")

    # ✅ **Vérification des colonnes finales avant normalisation**
    print(f"✅ Colonnes finales avant normalisation : {df.columns.tolist()}")

    # Réorganisation des colonnes
    df = df[selected_features]
    
    # Chargement du scaler et transformation
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    df[selected_features] = scaler.transform(df[selected_features])

    # 🔍 **Affichage des features après normalisation**
    print("🛠 Features après normalisation :")
    print(df.head())

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
        
        # Prédiction de probabilité
        y_pred_prob = best_model.predict_proba(features_processed)[:, 1]
        
        # 🔍 Vérification des probabilités prédites
        print(f"🔍 Probabilités des prédictions : {y_pred_prob}")

        # Application du seuil optimal
        print(f"🔍 Seuil utilisé : {best_threshold}")
        y_pred = (y_pred_prob < best_threshold).astype(int)
        decisions = ["Crédit accordé" if pred == 1 else "Crédit refusé" for pred in y_pred]
        
        response = {
            "probability": y_pred_prob.tolist(),
            "prediction": y_pred.tolist(),
            "decision": decisions
        }
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 5000 en fallback local
    app.run(host="0.0.0.0", port=port, debug=False)
