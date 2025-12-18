# 🗳️ Tunisian Election Sentiment Analysis MLOps Pipeline

[![CI/CD Pipeline](https://github.com/AlaDdin0709/mlops_election/actions/workflows/ci.yml/badge.svg)](https://github.com/AlaDdin0709/mlops_election/actions)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-purple)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)

## 📋 Project Overview
This project implements a complete **MLOps pipeline** for analyzing sentiment in Tunisian election-related comments. It leverages a modern stack to ensure reproducibility, automation, and scalability.

**Key Features:**
*   **Data Versioning:** Handled by **DVC** with storage on S3/DagsHub.
*   **Experiment Tracking:** Uses **MLflow** for logging metrics, parameters, and artifacts (models & vectorizers).
*   **CI/CD:** Fully automated pipeline via **GitHub Actions** (Training, Testing, Registration, Deployment).
*   **Deployment:**
    *   **Backend:** FastAPI for high-performance inference.
    *   **Frontend:** Streamlit for user interaction.
    *   **Infrastructure:** Deployed on **Azure Virtual Machine** via Docker Compose.

---

## 🏗️ Architecture

The pipeline consists of the following stages:

1.  **Data Ingestion:** Raw data is versioned with DVC.
2.  **Preprocessing:** Text cleaning and TF-IDF vectorization.
3.  **Training:**
    *   Multiple models trained (SVM, XGBoost, etc.).
    *   Best model selected based on **F1 Score**.
    *   Artifacts (Model + Vectorizer) logged to MLflow.
4.  **Registration:** The best model is automatically registered for production.
5.  **Deployment:**
    *   REST API (FastAPI) serves the model.
    *   Streamlit Dashboard consumes the API.

---

## 🛠️ Tech Stack

| Component | Tool | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.9 | Core programming language |
| **Version Control** | Git & GitHub | Code versioning |
| **Data Versioning** | DVC | Data lineage and versions |
| **Experiment Tracking** | MLflow | Metrics, parameters, artifact storage |
| **Storage** | DagsHub | Remote storage for DVC and MLflow |
| **CI/CD** | GitHub Actions | Automated workflow |
| **API** | FastAPI | Model serving interface |
| **Frontend** | Streamlit | Assessment UI |
| **Containerization** | Docker | Environment consistency |
| **Cloud** | Azure VM | Hosting infrastructure |

---

## 🚀 Setup & Installation

### Prerequisites
*   Python 3.9+
*   Docker & Docker Compose
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/AlaDdin0709/mlops_election.git
cd mlops_election
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/AlaDdin0709/mlops_election.mlflow
MLFLOW_EXPERIMENT=sentiment_classification_tunisian
DAGSHUB_REPO_NAME=mlops_election
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
```

### 4. Pull Data (DVC)
```bash
dvc pull -v
```

---

## 🏃 Usage

### Local Training
```bash
python scripts/train.py
```
This will:
*   Preprocess data.
*   Train multiple models.
*   Log results to MLflow.
*   Save the best model locally.

### Local API & Frontend (Docker)
```bash
docker-compose up --build
```
*   **API:** `http://localhost:8000/docs`
*   **Frontend:** `http://localhost:8501`

---

## 🔄 CI/CD Pipeline

We use **GitHub Actions** for automation.

### Workflow: `ci.yml`
1.  **Trigger:** Pushes to `master` branch.
2.  **CI Job:**
    *   Installs dependencies.
    *   Pulls data via DVC.
    *   Runs unit tests (`pytest`).
    *   Trains models (`train.py`).
    *   Builds Docker images.
3.  **CD Job (Deployment):**
    *   Downloads best model artifacts (`register_best_model.py`) from MLflow.
    *   Connects to Azure VM via SSH.
    *   Transfers artifacts (`model_registry/`) using SCP.
    *   Updates code and restarts containers (`docker-compose up -d`).

---

## 📊 Monitoring (In Progress)

We are integrating **Evidently AI** to monitor:
*   **Data Drift:** Detect changes in input data distribution.
*   **Model Performance:** Track metric degradation over time.
*   **Reporting:** Automated HTML reports generated in the pipeline.

---

## 👥 Authors
*   **AlaDdin0709** - *MLOps Engineer*

---

*"Building reliable ML systems with automation and best practices."*
