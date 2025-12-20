# 🗳️ RAPPORT INTÉGRAL MLOPS : ANALYSE DES ÉLECTIONS TUNISIENNES

Ce rapport documente la plateforme MLOps complète mise en place, du stockage des données sur **DagsHub** au déploiement final sur **Azure**, en passant par le monitoring sur **Arize AI**.

---

## 🏗️ 1. ARCHITECTURE ET INFRASTRUCTURE
Le projet repose sur une infrastructure multi-sites synchronisée.

### 🌐 DagsHub : Le Hub de Données et Expériences
*   **DVC Remote** : Héberge les versions réelles des fichiers Excel (`version1.xlsx`, `version2.xlsx`, etc.).
*   **MLflow Tracking** : Serveur centralisé pour comparer les performances des modèles entraînés.

### ☁️ Azure : Serveur de Production
*   Déploiement sur une **VM Azure** via **Docker Compose**.
*   **CD (Continuous Deployment)** : Mise à jour automatique de l'application via GitHub Actions lors de chaque push sur `master`.

---

## 📦 2. GESTION DES DONNÉES (DVC)
*   **Stratégie Delta** : Chaque nouvelle version de données (`v2`, `v3`) ne contient que les nouveaux commentaires collectés.
*   **Versionnement** : Garantie que chaque modèle est lié à une version précise des données.

---

## 🧠 3. ENTRAÎNEMENT ET SÉLECTION
*   **Modèles** : Comparaison entre SVM, Random Forest, XGBoost et Réseaux de Neurones.
*   **Optimisation** : Augmentation des paramètres d'entraînement (jusqu'à 1000 époques) et passage à 10 000 features pour une meilleure précision.
*   **Critère Champion** : Sélection automatique basée sur la **Précision (Precision)**.
*   **Prétraitement** : Nettoyage du dialecte tunisien et vectorisation TF-IDF.

---

## 🦅 4. MONITORING (ARIZE AI)
*   **Drift de Données** : Surveillance automatique de l'évolution du langage électoral.
*   **Indicateurs de Performance** : Suivi en temps réel de la précision du modèle en production.
*   **Gardiens (Monitors)** : Alertes automatiques en cas de baisse de performance ou de changement suspect dans les données.

---

## 🚀 5. LE PIPELINE AUTOMATISÉ (CONTINUOUS TRAINING)
C'est le point fort du projet. Nous avons bouclé le cycle :
1.  **Réception** : Ajout de nouvelles données.
2.  **Vérification** : Interrogation automatique d'Arize AI pour détecter un éventuel **Drift**.
3.  **Action** : Si un Drift est détecté, GitHub Actions lance le réentraînement sur l'ensemble accumulé (`v1 + v2 + ...`).
4.  **Mise à jour** : Le nouveau modèle "Champion" est déployé sur Azure sans intervention humaine.

---

**Le système est désormais une infrastructure MLOps robuste, capable de s'adapter seule aux changements de l'opinion publique tunisienne.**
