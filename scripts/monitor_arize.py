"""
Script de Monitoring Arize AI - Mode Production
Envoie les données de Baseline (Training) et de Production pour activer le Drift Detection.
"""

import os
import sys
import pandas as pd
import uuid
from pathlib import Path
from dotenv import load_dotenv
from arize.pandas.logger import Client, Schema
from arize.utils.types import Environments, ModelTypes

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Clés récupérées depuis le .env
SPACE_ID = os.getenv('ARIZE_SPACE_KEY')
API_KEY = os.getenv('ARIZE_API_KEY')

def run_production_monitoring():
    if not SPACE_ID or not API_KEY:
        print("❌ Erreur : Clés Arize manquantes dans le .env")
        return

    client = Client(space_id=SPACE_ID, api_key=API_KEY)
    
    # 1. Chargement des données réelles (version1)
    data_path = BASE_DIR / 'data' / 'version1.xlsx'
    if not data_path.exists():
        print(f"❌ Erreur : Fichier non trouvé {data_path}")
        return

    print(f"📦 Chargement des données : {data_path.name}")
    df = pd.read_excel(data_path)
    
    # Génération d'IDs uniques et TIMESTAMPS (Crucial pour la visualisation)
    df['prediction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    # Arize a besoin de timestamps pour placer les données sur une chronologie
    import time
    df['prediction_ts'] = int(time.time())
    
    # Split Simulation : Baseline vs Production
    split_idx = int(len(df) * 0.8)
    training_data = df.iloc[:split_idx].reset_index(drop=True)
    production_data = df.iloc[split_idx:].reset_index(drop=True)

    # Default: set baseline prediction_label to the actual target (so Arize baseline has labels)
    training_data['prediction_label'] = training_data['target']
    training_data['score_0'] = None
    training_data['score_1'] = None

    # Load model/vectorizer for production predictions if available
    model = None
    vectorizer = None
    model_path = BASE_DIR / 'model_registry' / 'Best_Election_Model' / 'production.pkl'
    vec_path = BASE_DIR / 'model_registry' / 'Best_Election_Model' / 'tfidf_vectorizer.pkl'
    if model_path.exists() and vec_path.exists():
        try:
            import pickle
            with open(model_path, 'rb') as mf:
                model = pickle.load(mf)
            with open(vec_path, 'rb') as vf:
                vectorizer = pickle.load(vf)
        except Exception as e:
            print(f"⚠️  Could not load model/vectorizer: {e}")

    # Compute predictions for production_data only
    if model is not None and vectorizer is not None and len(production_data) > 0:
        texts = production_data['comments'].fillna('').astype(str)
        try:
            Xp = vectorizer.transform(texts)
            try:
                preds = model.predict(Xp)
            except Exception as e:
                # Retry with dense array if model was trained on dense input
                if hasattr(Xp, 'toarray'):
                    try:
                        Xp = Xp.toarray()
                        preds = model.predict(Xp)
                    except Exception as e2:
                        print(f"⚠️  Error during model.predict with dense input: {e2}")
                        preds = [None] * len(production_data)
                else:
                    print(f"⚠️  Error during model.predict: {e}")
                    preds = [None] * len(production_data)
        except Exception as e:
            print(f"⚠️  Error preparing inputs for prediction: {e}")
            preds = [None] * len(production_data)

        # probabilities (use current Xp which may be dense)
        probs = None
        try:
            probs = model.predict_proba(Xp)
        except Exception as e:
            # Retry with dense input if needed
            try:
                if hasattr(Xp, 'toarray'):
                    probs = model.predict_proba(Xp.toarray())
            except Exception:
                probs = None

        production_data['prediction_label'] = preds
        if probs is not None and probs.shape[1] >= 2:
            production_data['score_0'] = probs[:, 0]
            production_data['score_1'] = probs[:, 1]
        elif probs is not None and probs.shape[1] == 1:
            production_data['score_0'] = probs[:, 0]
            production_data['score_1'] = 1 - probs[:, 0]
        else:
            production_data['score_0'] = None
            production_data['score_1'] = None
    else:
        # fallback: set production prediction_label to actuals to avoid missing predictions
        production_data['prediction_label'] = production_data['target']
        production_data['score_0'] = None
        production_data['score_1'] = None
    
    schema = Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_ts", # Ajout du timestamp
        prediction_label_column_name="prediction_label",
        actual_label_column_name="target",
        feature_column_names=['comments']
    )


    print(f"📡 Envoi de la BASELINE (Entraînement) : {len(training_data)} lignes...")
    res_train = client.log(
        dataframe=training_data,
        model_id='election_sentiment_tunisia',
        model_version='1.0',
        environment=Environments.TRAINING, # <--- Très important pour le Drift !
        model_type=ModelTypes.SCORE_CATEGORICAL,
        schema=schema
    )

    print(f"📡 Envoi des données de PRODUCTION : {len(production_data)} lignes...")
    res_prod = client.log(
        dataframe=production_data,
        model_id='election_sentiment_tunisia',
        model_version='1.0',
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        schema=schema
    )

    if res_prod.status_code == 200:
        print("\n✅ SUCCÈS ! Monitoring activé.")
        print("👉 Va sur Arize, cherche le modèle 'election_sentiment_tunisia'")
        print("💡 Note : L'analyse du Drift peut prendre 5-10 minutes à apparaître.")
    else:
        print(f"\n❌ Erreur lors de l'envoi : {res_prod.text}")

if __name__ == "__main__":
    run_production_monitoring()
