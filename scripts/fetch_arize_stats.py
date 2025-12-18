"""
Script pour récupérer les statistiques (Accuracy, Drift) depuis Arize via l'API GraphQL.
Cela permet de "lire" les résultats sans passer par l'interface web.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Clés
SPACE_ID = os.getenv('ARIZE_SPACE_KEY')
API_KEY = os.getenv('ARIZE_API_KEY')
# L'endpoint correct pour l'API publique
GRAPHQL_ENDPOINT = "https://app.arize.com/graphql"

def fetch_arize_metrics():
    if not SPACE_ID or not API_KEY:
        print("❌ Clés Arize manquantes dans le .env")
        return

    print(f"🔍 Interrogation d'Arize pour le modèle 'election_sentiment_tunisia'...")

    headers = {
        "x-api-key": API_KEY, # Utilisation de x-api-key au lieu d'authorization
        "content-type": "application/json",
    }

    # Requête finale corrigée (models est une Connection, on utilise edges > node)
    query = """
    query getModelStats($spaceId: ID!) {
      node(id: $spaceId) {
        ... on Space {
          name
          models {
            edges {
              node {
                name
                averagePerformanceMetric(metric: ACCURACY) {
                  value
                }
                driftStatus {
                  state
                }
                dataQualityStatus {
                  state
                }
              }
            }
          }
        }
      }
    }
    """
    
    variables = {
        "spaceId": SPACE_ID
    }

    try:
        response = requests.post(
            GRAPHQL_ENDPOINT,
            json={'query': query, 'variables': variables},
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'errors' in data:
                print(f"❌ Erreur GraphQL : {data['errors'][0]['message']}")
            else:
                space_data = data.get('data', {}).get('node', {})
                if space_data:
                    print(f"\n🏢 Espace : {space_data.get('name')}")
                    # On extrait les nodes depuis les edges
                    edges = space_data.get('models', {}).get('edges', [])
                    models = [e['node'] for e in edges if e.get('node')]
                    
                    # On cherche notre modèle spécifique
                    target_model_name = "election_sentiment_tunisia"
                    model = next((m for m in models if m['name'] == target_model_name), None)
                    
                    if model:
                        acc = model.get('averagePerformanceMetric')
                        drift = model.get('driftStatus', {}).get('state')
                        quality = model.get('dataQualityStatus', {}).get('state')
                        
                        print(f"\n🎯 Modèle : {model['name']}")
                        print(f"   📊 Accuracy (Dernière) : {acc['value'] if acc else 'N/A'}")
                        print(f"   📉 Statut Drift : {drift if drift else 'Calcul en cours...'}")
                        print(f"   🧹 Qualité Données : {quality if quality else 'N/A'}")
                    else:
                        model_names = [m['name'] for m in models]
                        print(f"❓ Modèle '{target_model_name}' non trouvé.")
                        print(f"   Modèles dispo : {', '.join(model_names) if model_names else 'Aucun'}")
                else:
                    print("❌ Espace non trouvé ou accès refusé.")
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Erreur lors de la requête : {e}")

if __name__ == "__main__":
    fetch_arize_metrics()
