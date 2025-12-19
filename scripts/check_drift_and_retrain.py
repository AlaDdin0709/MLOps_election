"""
Script de décision pour le Continuous Training (CT).
Vérifie le statut de Drift sur Arize et décide s'il faut réentraîner.
"""

import os
import sys
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

SPACE_ID = os.getenv('ARIZE_SPACE_KEY')
API_KEY = os.getenv('ARIZE_API_KEY')
GRAPHQL_ENDPOINT = "https://app.arize.com/graphql"

def check_drift_and_exit():
    if not SPACE_ID or not API_KEY:
        print("❌ Clés Arize manquantes. Skip retraining.")
        sys.exit(0)

    print("🔍 Analyse de la dérive (Drift) via Arize API...")
    
    query = """
    query getDriftStatus($spaceId: ID!) {
      node(id: $spaceId) {
        ... on Space {
          models(first: 50) {
            edges {
              node {
                name
                driftStatus {
                  status
                }
              }
            }
          }
        }
      }
    }
    """
    
    variables = {"spaceId": SPACE_ID}
    headers = {"x-api-key": API_KEY, "content-type": "application/json"}

    # Arize peut prendre du temps (10-15 min) pour traiter les données après un log.
    # On augmente la patience pour le premier run.
    retries = 6 # 6 essais
    wait_time = 180 # 3 minutes entre chaque essai (total 18 min)
    
    for i in range(retries):
        try:
            print(f"📡 Tentative {i+1}/{retries} : Vérification du statut sur Arize...")
            response = requests.post(GRAPHQL_ENDPOINT, json={'query': query, 'variables': variables}, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                space_data = data.get('data', {}).get('node', {})
                models = [e['node'] for e in space_data.get('models', {}).get('edges', [])]
                
                target_model = next((m for m in models if m['name'] == "election_sentiment_tunisia"), None)
                
                if target_model:
                    status = target_model.get('driftStatus', {}).get('status')
                    print(f"📊 Statut Drift actuel : {status}")
                    
                    if status in ["DEVIATING", "ACTIVE", "STABLE"]: # On accepte ces statuts
                        if status == "DEVIATING":
                            print("🚨 DRIFT DÉTECTÉ ! Déclenchement du réentraînement...")
                            sys.exit(1)
                        else:
                            print(f"✅ État sain ({status}). Pas de réentraînement nécessaire.")
                            sys.exit(0)
                
            print(f"⏳ Statut encore 'unmonitored' ou calcul en cours. Attente de {wait_time}s...")
            if i < retries - 1:
                time.sleep(wait_time)
        except Exception as e:
            print(f"⚠️ Erreur lors de la tentative {i+1}: {e}")
            if i < retries - 1:
                time.sleep(wait_time)

    print("⚠️ Arize n'a pas mis à jour le statut à temps. Skip training par sécurité.")
    sys.exit(0)

if __name__ == "__main__":
    check_drift_and_exit()
