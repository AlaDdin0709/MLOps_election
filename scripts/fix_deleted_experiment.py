"""
Fix deleted MLflow experiment - either restore or permanently delete it
"""
import os
from dotenv import load_dotenv
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Config
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', 'AlaDdin0709')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME', 'mlops_election')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
EXPERIMENT_NAME = "sentiment_classification_tunisian"

# Configure authentication
if DAGSHUB_TOKEN:
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print("="*80)
print("🔧 FIX DELETED EXPERIMENT")
print("="*80)
print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"Experiment: {EXPERIMENT_NAME}\n")

# Try to find the experiment (including deleted ones)
try:
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if exp is None:
        print(f"✅ Experiment '{EXPERIMENT_NAME}' doesn't exist")
        print(f"   Creating new experiment...")
        exp_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Created experiment with ID: {exp_id}")
    elif exp.lifecycle_stage == "deleted":
        print(f"⚠️  Experiment '{EXPERIMENT_NAME}' is DELETED")
        print(f"   Experiment ID: {exp.experiment_id}")
        print(f"\nChoose action:")
        print(f"   1. Restore experiment (keep all runs)")
        print(f"   2. Create new experiment with different name")
        
        choice = input("\nEnter 1 or 2: ").strip()
        
        if choice == "1":
            # Restore the experiment
            client.restore_experiment(exp.experiment_id)
            print(f"\n✅ Experiment '{EXPERIMENT_NAME}' restored!")
            print(f"   All previous runs are available again")
        else:
            # Use a different name
            new_name = f"{EXPERIMENT_NAME}_v2"
            print(f"\n📝 Creating new experiment: {new_name}")
            exp_id = mlflow.create_experiment(new_name)
            print(f"✅ Created experiment with ID: {exp_id}")
            print(f"\n⚠️  UPDATE YOUR .env FILE:")
            print(f"   MLFLOW_EXPERIMENT={new_name}")
    else:
        print(f"✅ Experiment '{EXPERIMENT_NAME}' is active")
        print(f"   Experiment ID: {exp.experiment_id}")
        print(f"   No action needed")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Done! You can now run: python scripts/train.py")
print("="*80)