import os
import shutil
import mlflow
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



def main():
    print(f"🔧 Configuring MLflow -> {MLFLOW_TRACKING_URI}")
    
    # Configure DagsHub authentication if token is available
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Find experiment
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found at {MLFLOW_TRACKING_URI}")
        return 1

    print(f"🔎 Searching runs in experiment '{EXPERIMENT_NAME}' (id={exp.experiment_id})")

    # Search for best run by f1_score (descending)
    try:
        df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.f1_score > 0",
            order_by=["metrics.f1_score DESC"],
            max_results=1
        )
    except Exception as e:
        print(f"❌ Error searching runs: {e}")
        return 1

    if df.empty:
        print("❌ No runs found matching criteria.")
        return 1

    best = df.iloc[0]
    run_id = best.run_id
    model_name_param = best.get('params.model_type') or best.get('params.model_name') or 'Election_Best'
    f1 = best.get('metrics.f1_score')
    print(f"🏆 Best run: {run_id}  model: {model_name_param}  f1_score: {f1}")

    # Prepare destination
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Attempt to download the artifact. Try common locations.
    downloaded = None
    candidates = ["model/model.pkl", "model.pkl", "model"]
    for art in candidates:
        try:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=art)
            downloaded = Path(local)
            print(f"⬇️  Downloaded artifact path: {downloaded}")
            break
        except Exception:
            continue

    if downloaded is None:
        print("❌ Could not download model artifact from MLflow. Tried common artifact paths.")
        return 1

    # If downloaded is a directory, search for a .pkl inside
    if downloaded.is_dir():
        pkl_files = list(downloaded.rglob('*.pkl'))
        if pkl_files:
            src = pkl_files[0]
        else:
            # try to find any file named model.*
            candidates_any = list(downloaded.rglob('model.*'))
            if candidates_any:
                src = candidates_any[0]
            else:
                print(f"❌ No .pkl file found inside downloaded artifact dir: {downloaded}")
                return 1
    else:
        src = downloaded

    # Copy to standardized production path
    try:
        shutil.copy2(src, DEST_PATH)
        print(f"✅ Copied model to {DEST_PATH}")
    except Exception as e:
        print(f"❌ Error copying model artifact: {e}")
        return 1

    # Optional: register model in MLflow Model Registry
    reg_model_name = os.getenv('REGISTER_MODEL_NAME', f"Election_{model_name_param}")
    model_uri = f"runs:/{run_id}/model"
    try:
        print(f"🔐 Registering model in MLflow registry as '{reg_model_name}' (optional)")
        mlflow.register_model(model_uri, reg_model_name)
    except Exception as e:
        print(f"⚠️  Registration warning: {e}")

    print("🎉 Done. You can now serve the model from model_registry/Best_Election_Model/production.pkl")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
