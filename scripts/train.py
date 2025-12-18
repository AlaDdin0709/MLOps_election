"""
Script d'entraînement des modèles avec MLflow - MLOps Election
Entraîne et évalue les modèles ML classiques et TunBERT
"""

import os
import sys
import argparse
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
from mlflow.models.signature import infer_signature
import dagshub

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Model registry for deployment artifacts
MODEL_REGISTRY_DIR = BASE_DIR / 'model_registry' / 'Best_Election_Model'
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

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
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')
    
    MLFLOW_TRACKING_URI = os.getenv(
        'MLFLOW_TRACKING_URI',
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    )
    
    print("\n🔧 Configuration MLflow & DagsHub")
    print("-" * 80)
    
    # Configure DagsHub authentication if token is available
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        print(f"✅ DagsHub authentication configured with token")
    else:
        # Only try dagshub.init() if no token (local development)
        try:
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
# Preprocessing will be done in-memory - no separate loading needed
# ============================================================================

# ============================================================================
# Modèles ML Classiques
# ============================================================================

def get_ml_models():
    """Retourne la configuration des modèles ML"""
    models = {
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
        'SVM_Linear': SVC(kernel='linear', random_state=42),
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
                
                # Log modèle: use model-specific artifact folder and record signature/input_example
                try:
                    artifact_name = f"Election_{model_name}"
                    # Prepare a small input example for signature inference
                    try:
                        X_sample = Xtr[:1]
                        if hasattr(X_sample, 'toarray'):
                            X_sample_df = pd.DataFrame(X_sample.toarray())
                        else:
                            X_sample_df = pd.DataFrame(X_sample)
                        y_sample_pred = model.predict(X_sample_df)
                        signature = infer_signature(X_sample_df, y_sample_pred)
                        input_example = X_sample_df.iloc[[0]]
                    except Exception:
                        signature = None
                        input_example = None

                    mlflow.sklearn.log_model(
                        model,
                        artifact_path=artifact_name,
                        registered_model_name=artifact_name,
                        signature=signature,
                        input_example=input_example
                    )
                except Exception as e:
                    print(f"⚠️  Warning: could not log model with signature: {e}")
                
                # Log confusion matrix as MLflow artifact (in-memory, no file)
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual_0', 'Actual_1'],
                    columns=['Pred_0', 'Pred_1']
                )
                # Log as text artifact to MLflow (let Python handle cleanup)
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                    tmp_path = tmp.name
                    cm_df.to_csv(tmp_path)
                
                try:
                    mlflow.log_artifact(tmp_path, artifact_path='confusion_matrices')
                finally:
                    # Ensure file is closed before deletion
                    try:
                        os.unlink(tmp_path)
                    except (PermissionError, OSError):
                        pass  # File will be cleaned up by OS eventually
                
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
    
    # Sauvegarder CSV dans model registry
    results_path = MODEL_REGISTRY_DIR / 'ml_models_results.csv'
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

        # CLI: optional custom data path
        parser = argparse.ArgumentParser()
        parser.add_argument('--data-path', '-d', help='Optional path to raw data Excel file (overrides default)')
        args = parser.parse_args()

        # Always run preprocessing in-memory using functions from scripts/preprocess.py
        # Ensure scripts folder is importable
        SCRIPTS_DIR = Path(__file__).resolve().parent
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))

        try:
            import preprocess
        except Exception as e:
            raise RuntimeError(f"Could not import preprocess module: {e}")

        # Determine data path
        data_path = Path(args.data_path) if args.data_path else (preprocess.DATA_DIR / 'version1.xlsx')
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Run preprocessing functions in-memory
        print("\n" + "="*80)
        print("🧪 PREPROCESSING - Integrated in Training Pipeline")
        print("="*80)
        df = preprocess.load_and_explore_data(data_path)
        df = preprocess.preprocess_text(df, text_col='comments')
        X, vectorizer = preprocess.vectorize_text(df, text_col='cleaned', max_features=5000)
        y = df['target'].values
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess.split_data(X, y, test_size=0.15, val_size=0.15, random_state=42)
        
        # Entraîner les modèles ML
        results, trained_models = train_ml_models(X_train, X_test, y_train, y_test)
        
        # Sauvegarder résumé
        df_results = save_results_summary(results)
        
        # Sauvegarder le vectorizer dans model_registry (co-located with production model)
        # Ensure subfolder exists
        best_model_dir = MODEL_REGISTRY_DIR / 'Best_Election_Model'
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        vectorizer_path = best_model_dir / 'tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"✅ Vectorizer sauvegardé: {vectorizer_path}")
        
        # Log vectorizer as artifact in the last active MLflow run
        if mlflow.active_run():
            mlflow.log_artifact(str(vectorizer_path))
            print(f"✅ Vectorizer loggé comme artifact MLflow")
        
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*80)
        print(f"📁 Model Registry: {MODEL_REGISTRY_DIR}")
        print(f"🔗 MLflow UI: Consultez DagsHub pour voir les runs")
        print(f"\n📦 Next step: python scripts/register_best_model.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
