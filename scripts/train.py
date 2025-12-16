"""
Script d'entraînement des modèles avec MLflow - MLOps Election
Entraîne et évalue les modèles ML classiques et TunBERT
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ML classiques
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from scipy.sparse import issparse

# MLflow & DagsHub
import mlflow
import mlflow.sklearn
import dagshub

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

PROCESSOR_DIR = BASE_DIR / 'processors'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# XGBoost (optionnel)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost non installé - modèle ignoré")

print("="*80)
print("🚀 ENTRAÎNEMENT DES MODÈLES - MLOps Election")
print("="*80)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Répertoire: {BASE_DIR}")
print()

# ============================================================================
# Configuration MLflow & DagsHub
# ============================================================================

def setup_mlflow():
    """Configure MLflow tracking avec DagsHub"""
    DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', 'AlaDdin0709')
    DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME', 'mlops_election')
    MLFLOW_TRACKING_URI = os.getenv(
        'MLFLOW_TRACKING_URI',
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    )
    
    print("🔧 Configuration MLflow & DagsHub")
    print("-" * 80)
    
    try:
        # Initialiser DagsHub
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
        print(f"✅ DagsHub initialisé: {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")
    except Exception as e:
        print(f"⚠️  DagsHub warning: {e}")
    
    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sentiment_classification_tunisian")
    
    print(f"✅ MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"✅ Experiment: sentiment_classification_tunisian")
    print()

# ============================================================================
# Chargement des données
# ============================================================================

def load_preprocessed_data():
    """Charge les données preprocessées"""
    print("📦 Chargement des données preprocessées")
    print("-" * 80)
    
    data_path = PROCESSOR_DIR / 'preprocessed_data.pkl'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Données preprocessées non trouvées: {data_path}\n"
            "Exécutez d'abord: python scripts/preprocess.py"
        )
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"✅ Données chargées:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    print()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_vectorizer():
    """Charge le vectorizer TF-IDF"""
    vectorizer_path = PROCESSOR_DIR / 'tfidf_vectorizer.pkl'
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✅ Vectorizer chargé: {len(vectorizer.get_feature_names_out())} features\n")
    return vectorizer

# ============================================================================
# Modèles ML Classiques
# ============================================================================

def get_ml_models():
    """Retourne la configuration des modèles ML"""
    models = {
        'Naive_Bayes': GaussianNB(),
        'Neural_Network': MLPClassifier(
            hidden_layer_sizes=(10, 10),
            activation='logistic',
            solver='adam',
            max_iter=500,
            random_state=42
        ),
        'Logistic_Regression': LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        ),
        'SVM_Linear': SVC(kernel='linear', random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42),
        'SVM_Sigmoid': SVC(kernel='sigmoid', random_state=42),
        'SVM_Poly': SVC(kernel='poly', degree=2, random_state=42)
    }
    
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
    
    return models

def calculate_metrics(y_true, y_pred):
    """Calcule toutes les métriques"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }

def train_ml_models(X_train, X_test, y_train, y_test):
    """Entraîne tous les modèles ML avec MLflow tracking"""
    print("🤖 ENTRAÎNEMENT DES MODÈLES ML CLASSIQUES")
    print("="*80)
    
    models = get_ml_models()
    results = []
    trained_models = {}
    failed_models = []
    
    for model_name, model in models.items():
        print(f"\n🔄 Entraînement: {model_name}")
        
        # Convertir sparse matrices si nécessaire
        try:
            Xtr = X_train.toarray() if issparse(X_train) else X_train
            Xte = X_test.toarray() if issparse(X_test) else X_test
        except Exception:
            Xtr, Xte = X_train, X_test
        
        # Démarrer un run MLflow
        with mlflow.start_run(run_name=f"ML_{model_name}"):
            try:
                # Entraînement
                start_time = time.time()
                model.fit(Xtr, y_train)
                training_time = time.time() - start_time
                
                # Prédiction
                y_pred = model.predict(Xte)
                
                # Métriques
                metrics = calculate_metrics(y_test, y_pred)
                
                # Log dans MLflow
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("algorithm", type(model).__name__)
                
                # Log hyperparams
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    for key, value in params.items():
                        if value is not None and not callable(value):
                            try:
                                mlflow.log_param(f"hp_{key}", value)
                            except:
                                pass
                
                # Log métriques
                mlflow.log_metric("accuracy", metrics['accuracy'])
                mlflow.log_metric("precision", metrics['precision'])
                mlflow.log_metric("recall", metrics['recall'])
                mlflow.log_metric("f1_score", metrics['f1_score'])
                mlflow.log_metric("training_time_seconds", training_time)
                
                # Log modèle
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"Election_{model_name}"
                )
                
                # Log confusion matrix comme artefact
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual_0', 'Actual_1'],
                    columns=['Pred_0', 'Pred_1']
                )
                cm_path = MODELS_DIR / f'cm_{model_name}.csv'
                cm_df.to_csv(cm_path)
                mlflow.log_artifact(str(cm_path))
                
                # Sauvegarder localement
                model_path = MODELS_DIR / f'model_{model_name.lower()}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                trained_models[model_name] = model
                results.append({
                    'Model': model_name,
                    **metrics,
                    'Training_Time': training_time
                })
                
                print(f"   ✅ Terminé - F1: {metrics['f1_score']:.4f} ({training_time:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                failed_models.append(model_name)
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
    
    if failed_models:
        print(f"\n⚠️  Modèles échoués: {failed_models}")
    
    return results, trained_models

# ============================================================================
# Résumé et export
# ============================================================================

def save_results_summary(results):
    """Sauvegarde le résumé des résultats"""
    print("\n📊 RÉSUMÉ DES RÉSULTATS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('f1_score', ascending=False)
    
    print(df_results.to_string(index=False))
    
    # Sauvegarder CSV
    results_path = MODELS_DIR / 'ml_models_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"\n✅ Résultats sauvegardés: {results_path}")
    
    # Meilleur modèle
    best = df_results.iloc[0]
    print(f"\n🏆 MEILLEUR MODÈLE: {best['Model']}")
    print(f"   F1-Score: {best['f1_score']:.4f}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    
    return df_results

# ============================================================================
# Main
# ============================================================================

def main():
    """Pipeline principal d'entraînement"""
    try:
        # Configurer MLflow
        setup_mlflow()
        
        # Charger les données
        X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
        vectorizer = load_vectorizer()
        
        # Entraîner les modèles ML
        results, trained_models = train_ml_models(X_train, X_test, y_train, y_test)
        
        # Sauvegarder résumé
        df_results = save_results_summary(results)
        
        # Sauvegarder le vectorizer
        vectorizer_path = MODELS_DIR / 'tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"✅ Vectorizer sauvegardé: {vectorizer_path}")
        
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*80)
        print(f"📁 Modèles: {MODELS_DIR}")
        print(f"🔗 MLflow UI: Consultez DagsHub pour voir les runs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
