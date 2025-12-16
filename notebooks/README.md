# 📓 Notebooks - Classification des Opinions Électorales Tunisiennes

## 📋 Organisation des Notebooks

Ce projet suit une organisation modulaire inspirée des best practices MLOps pour faciliter la reproductibilité et le versioning.

---

## 🗂️ Structure

```
notebooks/
├── preprocessing.ipynb          # Pipeline de préparation des données
├── modeling.ipynb               # Entraînement et évaluation des modèles
├── Tunbert_Classification.ipynb # [LEGACY] Notebook original TunBERT
├── Sentiment_Classification.ipynb # [LEGACY] Notebook original ML
└── README.md                    # Ce fichier
```

---

## 📖 Description des Notebooks

### 1️⃣ `preprocessing.ipynb` ✅ **NOUVEAU - À UTILISER**

**Objectif:** Pipeline complet de préparation des données

**Contenu:**
- ✅ Chargement du dataset (`data/version1.xlsx`)
- ✅ Exploration et analyse (EDA)
- ✅ Nettoyage du texte arabe (dialecte tunisien)
- ✅ Analyse de fréquence des mots
- ✅ Vectorisation TF-IDF avec stopwords tunisiens
- ✅ Split stratifié Train/Val/Test (70/15/15)
- ✅ Sauvegarde des artefacts dans `processors/`

**Outputs:**
```
processors/
├── preprocessed_data.pkl      # Données train/val/test vectorisées
├── tfidf_vectorizer.pkl       # Vectorizer TF-IDF ajusté
└── cleaned_texts.pkl          # Textes originaux et nettoyés
```

**Exécution:**
1. Ouvrir `preprocessing.ipynb`
2. Exécuter toutes les cellules séquentiellement
3. Vérifier la création du dossier `processors/`

---

### 2️⃣ `modeling.ipynb` ✅ **NOUVEAU - À UTILISER**

**Objectif:** Entraînement, évaluation et comparaison des modèles

**Contenu:**

**A. Modèles ML Classiques (TF-IDF):**
- Naive Bayes
- Neural Network (MLP)
- SVM Linear
- SVM RBF
- SVM Sigmoid
- SVM Polynomial

**B. Transformer (TunBERT):**
- Fine-tuning du modèle pré-entraîné `tunis-ai/TunBERT`
- Entraînement sur GPU (si disponible)
- Évaluation complète

**Outputs:**
```
models/
├── model_naive_bayes.pkl
├── model_neural_network.pkl
├── model_svm_linear.pkl
├── model_svm_rbf.pkl
├── model_svm_sigmoid.pkl
├── model_svm_poly.pkl
├── tfidf_vectorizer.pkl
├── tunbert_final_model/        # Modèle TunBERT + tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
├── models_comparison.csv        # Tableau comparatif
└── models_comparison.png        # Graphiques
```

**Exécution:**
1. **Prérequis:** Exécuter d'abord `preprocessing.ipynb`
2. Ouvrir `modeling.ipynb`
3. Exécuter toutes les cellules séquentiellement
4. Consulter les résultats dans `models/`

---

### 3️⃣ Notebooks Legacy (Anciens) ⚠️

#### `Tunbert_Classification.ipynb` ⚠️ **LEGACY**
- Notebook monolithique original pour TunBERT
- **Ne plus utiliser** - remplacé par `preprocessing.ipynb` + `modeling.ipynb`
- Conservé pour référence historique

#### `Sentiment_Classification.ipynb` ⚠️ **LEGACY**
- Notebook monolithique original pour ML classiques
- **Ne plus utiliser** - remplacé par `preprocessing.ipynb` + `modeling.ipynb`
- Conservé pour référence historique

---

## 🚀 Workflow Complet

### Étape 1: Preprocessing
```bash
# Ouvrir preprocessing.ipynb et exécuter toutes les cellules
# Outputs: processors/preprocessed_data.pkl, tfidf_vectorizer.pkl, cleaned_texts.pkl
```

### Étape 2: Modeling
```bash
# Ouvrir modeling.ipynb et exécuter toutes les cellules
# Outputs: models/*.pkl, tunbert_final_model/, comparison files
```

### Étape 3: Analyse des Résultats
```bash
# Consulter models/models_comparison.csv pour le tableau comparatif
# Consulter models/models_comparison.png pour les graphiques
```

---

## 📊 Artefacts Générés

### Preprocessing (`processors/`)
| Fichier | Description | Taille |
|---------|-------------|--------|
| `preprocessed_data.pkl` | Données train/val/test + métadonnées | ~XX MB |
| `tfidf_vectorizer.pkl` | Vectorizer TF-IDF ajusté | ~XX MB |
| `cleaned_texts.pkl` | Textes nettoyés + labels | ~XX MB |

### Modeling (`models/`)
| Fichier | Description | F1-Score |
|---------|-------------|----------|
| `model_naive_bayes.pkl` | Naive Bayes | TBD |
| `model_neural_network.pkl` | MLP | TBD |
| `model_svm_linear.pkl` | SVM Linear | TBD |
| `model_svm_rbf.pkl` | SVM RBF | TBD |
| `model_svm_sigmoid.pkl` | SVM Sigmoid | TBD |
| `model_svm_poly.pkl` | SVM Polynomial | TBD |
| `tunbert_final_model/` | TunBERT fine-tuné | TBD |
| `models_comparison.csv` | Résultats comparatifs | - |

---

## 🔧 Dépendances

```bash
# Environnement Python recommandé
pip install pandas numpy scikit-learn
pip install torch transformers datasets
pip install matplotlib seaborn
pip install openpyxl  # Pour lire .xlsx
```

---

## 📝 Notes Importantes

1. **Ordre d'exécution:** Toujours exécuter `preprocessing.ipynb` AVANT `modeling.ipynb`

2. **GPU pour TunBERT:** 
   - L'entraînement TunBERT utilisera automatiquement le GPU si disponible
   - Sans GPU, l'entraînement prendra plus de temps

3. **Versioning DVC:**
   - Les dossiers `processors/` et `models/` doivent être trackés avec DVC
   - Ne pas commit ces dossiers dans Git

4. **Reproductibilité:**
   - Tous les random seeds sont fixés (`random_state=42`)
   - Les résultats doivent être reproductibles

---

## 🎯 Prochaines Étapes MLOps

- [ ] Ajouter DVC tracking pour `data/`, `processors/`, `models/`
- [ ] Créer `dvc.yaml` pipeline
- [ ] Intégrer MLflow pour le tracking des expérimentations
- [ ] Ajouter DeepChecks pour la validation des données
- [ ] Configurer GitHub Actions pour CI/CD
- [ ] Déployer le meilleur modèle (API)

---

## 📧 Contact

Pour toute question sur l'organisation des notebooks, consulter le README principal du projet.
