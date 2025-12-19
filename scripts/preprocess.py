"""
Script de prétraitement des données - MLOps Election
Charge, nettoie et vectorise les données textuelles en dialecte tunisien
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# Stopwords tunisiens
tunisian_stopwords = [
    'في', 'من', 'الى', 'على', 'هذا', 'هذه', 'ذلك', 'تلك', 'هو', 'هي',
    'نحن', 'هم', 'انت', 'انتم', 'انتي', 'انا', 'هما', 'كان', 'كانت',
    'يكون', 'تكون', 'ليس', 'ليست', 'ما', 'لا', 'لم', 'لن', 'ان', 'اذا',
    'كل', 'بعض', 'عند', 'بعد', 'قبل', 'اثناء', 'خلال', 'منذ', 'حتى',
    'مع', 'بدون', 'ضد', 'عن', 'الي', 'اللي', 'الى', 'إلى', 'علي', 'على',
    'هاذي', 'هاذا', 'هكا', 'هكة', 'برشا', 'ياسر', 'شوية', 'زادة', 'كيما',
    'باهي', 'موش', 'ماهو', 'ماهي', 'مانيش', 'مانا', 'كانش', 'ماكانش',
    'والله', 'يزي', 'معناها', 'يعني', 'برا', 'توا', 'توة'
]

def clean_arabic_text(text):
    """
    Nettoie le texte arabe/tunisien
    """
    if pd.isna(text):
        return ""
    
    # Convertir en string
    text = str(text)
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les mentions (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Supprimer les hashtags (#tag)
    text = re.sub(r'#\w+', '', text)
    
    # Supprimer les emojis et symboles spéciaux
    text = re.sub(r'[^\u0600-\u06FF\s\w]', ' ', text)
    
    # Normaliser les caractères arabes
    text = re.sub(r'[أإآا]', 'ا', text)
    text = re.sub(r'[ىي]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    
    # Supprimer les chiffres
    text = re.sub(r'\d+', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Nettoyer les espaces au début et à la fin
    text = text.strip()
    
    return text

def load_and_explore_data(filepath):
    """
    Charge et explore les données
    """
    print("\n📦 ÉTAPE 1: Chargement des données")
    print("-" * 80)
    
    df = pd.read_excel(filepath)
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"   Colonnes: {list(df.columns)}")
    
    # Vérifier les valeurs manquantes
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Valeurs manquantes:\n{missing[missing > 0]}")
    
    # Distribution des classes
    if 'target' in df.columns:
        print(f"\n📊 Distribution des classes:")
        print(df['target'].value_counts())
        print(f"   Ratio: {df['target'].value_counts(normalize=True).to_dict()}")
    
    return df

def preprocess_text(df, text_col='comments'):
    """
    Nettoie les textes
    """
    print("\n🧹 ÉTAPE 2: Nettoyage des textes")
    print("-" * 80)
    
    # Appliquer le nettoyage
    print("   Nettoyage en cours...")
    df['cleaned'] = df[text_col].apply(clean_arabic_text)
    
    # Supprimer les lignes vides après nettoyage
    before = len(df)
    df = df[df['cleaned'].str.len() > 0].copy()
    after = len(df)
    
    if before != after:
        print(f"   ⚠️  {before - after} lignes vides supprimées")
    
    print(f"✅ Nettoyage terminé: {len(df)} textes")
    print(f"   Longueur moyenne: {df['cleaned'].str.len().mean():.1f} caractères")
    
    return df

def vectorize_text(df, text_col='cleaned', max_features=5000):
    """
    Vectorisation TF-IDF
    """
    print("\n🔢 ÉTAPE 3: Vectorisation TF-IDF")
    print("-" * 80)
    
    # Créer le vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=tunisian_stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Fit et transform
    X = vectorizer.fit_transform(df[text_col])
    
    print(f"✅ Vectorisation terminée")
    print(f"   Shape: {X.shape}")
    print(f"   Features: {len(vectorizer.get_feature_names_out())}")
    print(f"   Sparsité: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
    
    return X, vectorizer

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split stratifié 70-15-15
    """
    print("\n✂️  ÉTAPE 4: Split des données (70-15-15)")
    print("-" * 80)
    
    # Train/temp split (70/30)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(test_size + val_size), 
        random_state=random_state,
        stratify=y
    )
    
    # Val/test split (15/15)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"✅ Split terminé:")
    print(f"   Train: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"   Val:   {X_val.shape[0]} ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"   Test:  {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    
    # Vérifier la distribution des classes
    print(f"\n   Distribution Train: {np.bincount(y_train)}")
    print(f"   Distribution Val:   {np.bincount(y_val)}")
    print(f"   Distribution Test:  {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Note: This script is now imported by train.py for in-memory preprocessing
# No need to save artifacts to disk - train.py handles that

def main():
    """
    Pipeline principal de prétraitement
    NOTE: This script is now primarily imported by train.py for in-memory preprocessing.
    Running it standalone will only validate the preprocessing pipeline without saving artifacts.
    """
    try:
        print("="*80)
        print("🧪 PREPROCESSING VALIDATION ")
        print("="*80)
        
        # Charger les données
        data_path = DATA_DIR / 'version1.xlsx'
        if not data_path.exists():
            raise FileNotFoundError(f"Fichier de données non trouvé: {data_path}")
        
        df = load_and_explore_data(data_path)
        
        # Prétraiter les textes
        df = preprocess_text(df, text_col='comments')
        
        # Vectorisation TF-IDF
        X, vectorizer = vectorize_text(df, text_col='cleaned', max_features=5000)
        
        # Extraire les labels
        y = df['target'].values
        
        # Split des données
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.15, val_size=0.15, random_state=42
        )
        
        print("\n" + "="*80)
        print("✅ PREPROCESSING VALIDATION SUCCESSFUL!")
        print("="*80)
        print(f"📊 Data shapes validated:")
        print(f"   Train: {X_train.shape}")
        print(f"   Val:   {X_val.shape}")
        print(f"   Test:  {X_test.shape}")
        print(f"\n💡 To train models, run: python scripts/train.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
