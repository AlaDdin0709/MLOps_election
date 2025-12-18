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

    try:
        # Arize peut prendre du temps à mettre à jour le statut après un log.
        # En CI, on peut vouloir attendre un peu ou simplement vérifier l'état actuel.
        response = requests.post(GRAPHQL_ENDPOINT, json={'query': query, 'variables': variables}, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            space_data = data.get('data', {}).get('node', {})
            models = [e['node'] for e in space_data.get('models', {}).get('edges', [])]
            
            target_model = next((m for m in models if m['name'] == "election_sentiment_tunisia"), None)
            
            if target_model:
                status = target_model.get('driftStatus', {}).get('status')
                print(f"📊 Statut Drift actuel : {status}")
                
                # Logique de décision
                if status == "DEVIATING":
                    print("🚨 DRIFT DÉTECTÉ ! Déclenchement du réentraînement...")
                    sys.exit(1) # Code 1 -> Indique à la CI de lancer train.py
                else:
                    print("✅ Pas de dérive critique. Le modèle actuel reste en place.")
                    sys.exit(0)
            else:
                print("⚠️ Modèle non trouvé sur Arize. On continue sans réentraînement.")
                sys.exit(0)
        else:
            print(f"⚠️ Erreur API Arize ({response.status_code}). Skip retraining.")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        sys.exit(0)

if __name__ == "__main__":
    check_drift_and_exit()
