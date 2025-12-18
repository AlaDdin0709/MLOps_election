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

    # Requête pour obtenir les infos de base et l'ID de la baseline
    query = """
    query getModelStats($spaceId: ID!) {
      node(id: $spaceId) {
        ... on Space {
          name
          models(first: 50) {
            edges {
              node {
                id
                name
                modelPrimaryBaseline {
                  id
                }
                driftStatus {
                  status
                }
                dataQualityStatus {
                  status
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
                # Si erreur de schéma, on peut suggérer de vérifier le dashboard
            else:
                space_data = data.get('data', {}).get('node', {})
                if space_data:
                    print(f"\n🏢 Espace : {space_data.get('name')}")
                    edges = space_data.get('models', {}).get('edges', [])
                    models = [e['node'] for e in edges if e.get('node')]
                    
                    target_model_name = "election_sentiment_tunisia"
                    model = next((m for m in models if m['name'] == target_model_name), None)
                    
                    if model:
                        drift = model.get('driftStatus', {}).get('status')
                        quality = model.get('dataQualityStatus', {}).get('status')
                        baseline_id = model.get('modelPrimaryBaseline', {}).get('id') if model.get('modelPrimaryBaseline') else 'Aucune'
                        
                        print(f"\n🎯 Modèle : {model['name']}")
                        print(f"   🆔 ID Baseline : {baseline_id}")
                        print(f"   📉 Statut Drift : {drift if drift else 'Calcul en cours...'}")
                        print(f"   🧹 Qualité Données : {quality if quality else 'N/A'}")
                        
                        if drift == "DEVIATING":
                            print("\n🚨 ALERTE : Dérive détectée ! Un réentraînement est conseillé.")
                        else:
                            print("\n✅ État stable. Pas de dérive critique détectée.")
                            
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
