"""
Download best model and vectorizer from MLflow to local model_registry.
"""
import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Config (match training script)
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', '')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME', 'mlops_election')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT', 'sentiment_classification_tunisian')

# Destination for local registry
DEST_DIR = BASE_DIR / 'model_registry' / 'Best_Election_Model'
DEST_PATH = DEST_DIR / 'production.pkl'


def list_artifacts_recursive(client, run_id, path=""):
    """Recursively list all artifacts in a run (fast API calls, no downloads)."""
    artifacts = []
    try:
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                artifacts.extend(list_artifacts_recursive(client, run_id, item.path))
            else:
                artifacts.append(item.path)
    except Exception as e:
        print(f"⚠️  Error listing artifacts at path '{path}': {e}")
    return artifacts


def main():
    print(f"🔧 Configuring MLflow -> {MLFLOW_TRACKING_URI}")
    
    # Configure DagsHub authentication if token is available
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Find experiment
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found at {MLFLOW_TRACKING_URI}")
        return 1

    print(f"🔎 Searching runs in experiment '{EXPERIMENT_NAME}' (id={exp.experiment_id})")

    # Search for best runs by f1_score (descending), with start_time as tiebreaker
    try:
        df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.f1_score > 0",
            order_by=["metrics.f1_score DESC", "start_time DESC"],  # Best F1, then latest
            max_results=5  # Get top 5 to have fallback options
        )
    except Exception as e:
        print(f"❌ Error searching runs: {e}")
        return 1

    if df.empty:
        print("❌ No runs found matching criteria.")
        return 1

    # Prepare destination
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try runs in order until we find one with downloadable artifacts
    for idx, row in df.iterrows():
        run_id = row['run_id']
        model_name_param = row.get('params.model_type') or row.get('params.model_name') or 'Unknown'
        f1 = row.get('metrics.f1_score', 0)
        print(f"\n🏆 Trying run: {run_id}  model: {model_name_param}  f1_score: {f1:.4f}")

        # List all artifacts in this run (fast API call - no download)
        print("📋 Listing artifacts in run (no download yet)...")
        all_artifacts = list_artifacts_recursive(client, run_id)
        
        if not all_artifacts:
            print(f"⚠️  No artifacts found in run {run_id}, trying next...")
            continue
            
        print(f"   Found {len(all_artifacts)} artifact(s):")
        for art in all_artifacts[:15]:
            print(f"      - {art}")
        if len(all_artifacts) > 15:
            print(f"      ... and {len(all_artifacts) - 15} more")

        # Find model artifact - look for .pkl file in Election_* or model paths
        pkl_artifacts = [a for a in all_artifacts if a.endswith('.pkl')]
        model_pkl_files = [a for a in pkl_artifacts if 'model' in a.lower() and 'vectorizer' not in a.lower()]
        
        # Preferred: Election_<model>/model.pkl
        model_artifact_path = None
        preferred_pattern = f"Election_{model_name_param}/model.pkl"
        if preferred_pattern in all_artifacts:
            model_artifact_path = preferred_pattern
        elif model_pkl_files:
            model_artifact_path = model_pkl_files[0]
        
        if not model_artifact_path:
            print(f"⚠️  No model .pkl artifact found in run {run_id}, trying next...")
            continue

        print(f"⬇️  Downloading model from: {model_artifact_path}")
        try:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_artifact_path)
            local_path = Path(local_path)
            
            # Handle if directory was downloaded
            if local_path.is_dir():
                pkl_files = list(local_path.rglob('*.pkl'))
                src = pkl_files[0] if pkl_files else None
            else:
                src = local_path
            
            if src is None:
                print(f"⚠️  No .pkl in downloaded path {local_path}")
                continue
                
            shutil.copy2(src, DEST_PATH)
            print(f"✅ Model copied to {DEST_PATH}")
        except Exception as e:
            print(f"⚠️  Failed to download model: {e}")
            continue

        # Find and download vectorizer
        vec_artifacts = [a for a in all_artifacts if 'vectorizer' in a.lower() and a.endswith('.pkl')]
        if vec_artifacts:
            vec_artifact_path = vec_artifacts[0]
            print(f"⬇️  Downloading vectorizer from: {vec_artifact_path}")
            try:
                vec_local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=vec_artifact_path)
                vec_dest = DEST_DIR / 'tfidf_vectorizer.pkl'
                shutil.copy2(vec_local, vec_dest)
                print(f"✅ Vectorizer copied to {vec_dest}")
            except Exception as e:
                print(f"⚠️  Failed to download vectorizer: {e}")
        else:
            print("⚠️  No vectorizer artifact found in this run.")
            print("   The API will still work if train.py saved vectorizer locally.")

        # Optional: register model in MLflow Model Registry
        reg_model_name = os.getenv('REGISTER_MODEL_NAME', f"Election_{model_name_param}")
        # Use the artifact directory (without /model.pkl) for registration
        model_dir = model_artifact_path.rsplit('/', 1)[0] if '/' in model_artifact_path else model_artifact_path
        model_uri = f"runs:/{run_id}/{model_dir}"
        try:
            print(f"🔐 Registering model in MLflow registry as '{reg_model_name}'")
            mlflow.register_model(model_uri, reg_model_name)
        except Exception as e:
            print(f"⚠️  Registration warning: {e}")

        print("\n🎉 Done! Model ready at model_registry/Best_Election_Model/production.pkl")
        return 0

    print("\n❌ Could not download artifacts from any of the top runs.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
