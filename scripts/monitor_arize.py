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
# Prefer ARIZE_SPACE_ID; fallback to deprecated ARIZE_SPACE_KEY
SPACE_ID = os.getenv('ARIZE_SPACE_ID') or os.getenv('ARIZE_SPACE_KEY')
API_KEY = os.getenv('ARIZE_API_KEY')

# If only the deprecated variable exists in the environment, set ARIZE_SPACE_ID
# in-process so the Arize SDK reads the canonical var and does not emit a
# deprecation warning from environment inspection.
if not os.getenv('ARIZE_SPACE_ID') and os.getenv('ARIZE_SPACE_KEY'):
    os.environ['ARIZE_SPACE_ID'] = os.getenv('ARIZE_SPACE_KEY')
    try:
        # remove deprecated key from process env so SDK won't detect it
        del os.environ['ARIZE_SPACE_KEY']
    except Exception:
        pass

def run_production_monitoring():
    if not SPACE_ID or not API_KEY:
        print("❌ Erreur : Clés Arize manquantes dans le .env")
        return

    client = Client(space_id=SPACE_ID, api_key=API_KEY)

    # On cherche automatiquement le dernier fichier de version s'il n'est pas spécifié
    prod_data_path = None
    if len(sys.argv) > 1:
        prod_data_path = Path(sys.argv[1])
    else:
        # Cherche tous les fichiers version*.xlsx et prend le plus récent (trié par nom)
        version_files = sorted(list(BASE_DIR.glob('data/version*.xlsx')))
        if len(version_files) > 1:
            prod_data_path = version_files[-1] # Le dernier (ex: version2, version3...)
            print(f"🔎 Détection automatique : Dernier fichier trouvé -> {prod_data_path.name}")

    # 1. Chargement de la BASELINE (Toujours version1 par défaut pour le comparatif)
    baseline_path = BASE_DIR / 'data' / 'version1.xlsx'
    if not baseline_path.exists():
        print(f"❌ Erreur : Fichier baseline non trouvé {baseline_path}")
        return

    print(f"📦 Chargement de la Baseline : {baseline_path.name}")
    df_base = pd.read_excel(baseline_path)
    df_base['prediction_id'] = [str(uuid.uuid4()) for _ in range(len(df_base))]
    import time
    df_base['prediction_ts'] = int(time.time()) - 86400 # Simule hier pour la baseline

    # 2. Chargement de la PRODUCTION
    if prod_data_path and prod_data_path.exists() and prod_data_path != baseline_path:
        print(f"📦 Chargement de la Production : {prod_data_path.name}")
        production_data = pd.read_excel(prod_data_path)
    else:
        print("💡 Simulation via split de la baseline (car pas d'autre fichier version trouvé)...")
        split_idx = int(len(df_base) * 0.8)
        production_data = df_base.iloc[split_idx:].copy().reset_index(drop=True)
        df_base = df_base.iloc[:split_idx].copy().reset_index(drop=True)

    training_data = df_base
    baseline_data = training_data # Alias pour plus de clarté
    production_data['prediction_id'] = [str(uuid.uuid4()) for _ in range(len(production_data))]
    production_data['prediction_ts'] = int(time.time())

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

    def add_predictions(target_df, desc):
        if model is not None and vectorizer is not None and len(target_df) > 0:
            print(f"🧠 Génération des prédictions pour {desc}...")
            texts = target_df['comments'].fillna('').astype(str)
            try:
                Xp = vectorizer.transform(texts)
                if hasattr(Xp, 'toarray'):
                    Xp = Xp.toarray()
                
                target_df['prediction_label'] = model.predict(Xp)
                
                # Correction SVC : Certains modèles n'ont pas predict_proba si probas non activées à l'entraînement
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(Xp)
                    if probs is not None and probs.shape[1] >= 2:
                        target_df['score_0'] = probs[:, 0]
                        target_df['score_1'] = probs[:, 1]
                    else:
                        target_df['score_0'] = None
                        target_df['score_1'] = None
                else:
                    print(f"⚠️  Note : Le modèle {desc} ne supporte pas predict_proba. Scores ignorés.")
                    target_df['score_0'] = None
                    target_df['score_1'] = None
            except Exception as e:
                print(f"⚠️  Erreur prédiction {desc}: {e}")
                target_df['prediction_label'] = target_df['target']
                target_df['score_0'] = None
                target_df['score_1'] = None
        else:
            target_df['prediction_label'] = target_df['target']
            target_df['score_0'] = None
            target_df['score_1'] = None

    # Ajouter les prédictions aux deux datasets
    add_predictions(baseline_data, "Baseline")
    add_predictions(production_data, "Production")

    # Optionally generate embeddings for semantic drift visualization
    # Wrapped in try/except because EmbeddingGenerator API varies across Arize versions
    use_embeddings = False
    try:
        from arize.pandas.embeddings import EmbeddingGenerator
        from arize.utils.types import EmbeddingColumnNames

        print("🧠 Génération des Embeddings NLP (cela peut prendre un moment)...")
        
        # Try different use_case values depending on Arize SDK version
        generator = None
        for use_case in ["NLP", "nlp", "text_classification", "classification"]:
            try:
                generator = EmbeddingGenerator.from_use_case(
                    use_case=use_case,
                    model_name="distilbert-base-uncased",
                    tokenizer_max_length=512,
                    batch_size=100
                )
                break
            except (ValueError, TypeError):
                continue
        
        if generator is None:
            # Fallback: try default constructor if from_use_case doesn't work
            try:
                generator = EmbeddingGenerator(
                    model_name="distilbert-base-uncased",
                    tokenizer_max_length=512,
                    batch_size=100
                )
            except Exception:
                raise RuntimeError("Could not initialize EmbeddingGenerator")
        
        baseline_data = generator.generate_embeddings(
            dataframe=baseline_data,
            data_column_name="comments",
            embedding_name="comments_embedding"
        )
        production_data = generator.generate_embeddings(
            dataframe=production_data,
            data_column_name="comments",
            embedding_name="comments_embedding"
        )
        use_embeddings = True
        print("✅ Embeddings générés avec succès!")
        
    except Exception as e:
        print(f"⚠️ Embeddings non disponibles : {e}")
        print("💡 Envoi des données sans visualisation d'embeddings (drift features toujours actif).")
    
    # Build schema with or without embeddings
    if use_embeddings:
        from arize.utils.types import EmbeddingColumnNames
        schema = Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_ts",
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="score_1",
            actual_label_column_name="target",
            feature_column_names=['comments'],
            embedding_feature_column_names=[
                EmbeddingColumnNames(
                    vector_column_name="comments_embedding_vector",
                    data_column_name="comments",
                ),
            ]
        )
    else:
        schema = Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_ts",
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="score_1",
            actual_label_column_name="target",
            feature_column_names=['comments']
        )

    print(f"📡 Envoi de la BASELINE (Entraînement) : {len(baseline_data)} lignes...")
    # Pour la baseline, on envoie tout (features, predictions, actuals)
    res_train = client.log(
        dataframe=baseline_data,
        model_id='election_sentiment_tunisia',
        model_version='1.0',
        environment=Environments.TRAINING,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        schema=schema
    )

    print(f"📡 Envoi des données de PRODUCTION : {len(production_data)} lignes...")
    # Pour la production, on envoie les predictions et les features
    res_prod = client.log(
        dataframe=production_data,
        model_id='election_sentiment_tunisia',
        model_version='1.0',
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        schema=schema
    )

    # Diagnostic: always print responses for both requests to aid troubleshooting
    try:
        print(f"🔁 Arize baseline response: status={res_train.status_code}")
        if hasattr(res_train, 'text'):
            print(f"🔁 baseline text: {res_train.text}")
    except Exception:
        pass

    try:
        print(f"🔁 Arize production response: status={res_prod.status_code}")
        if hasattr(res_prod, 'text'):
            print(f"🔁 production text: {res_prod.text}")
    except Exception:
        pass

    if getattr(res_prod, 'status_code', None) == 200 and getattr(res_train, 'status_code', None) in (200, 201):
        print("\n✅ SUCCÈS ! Monitoring activé.")
        print("👉 Va sur Arize, cherche le modèle 'election_sentiment_tunisia'")
        print("💡 Note : L'analyse du Drift peut prendre 5-10 minutes à apparaître.")
    else:
        print("\n❌ Un ou plusieurs envois ont échoué. Vérifiez les réponses ci-dessus et vos clés Arize.")

if __name__ == "__main__":
    run_production_monitoring()
