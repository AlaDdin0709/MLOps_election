import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# API Configuration
API_URL = os.getenv("API_URL", "http://backend:8000" if os.path.exists("/.dockerenv") else "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="Analyse de Sentiment Tunisien | Élections",
    page_icon="🇹🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Tunisian theme colors */
    :root {
        --tunisian-red: #E70013;
        --tunisian-white: #FFFFFF;
        --dark-red: #8B0000;
        --gold: #FFD700;
        --dark-text: #1a1a1a;
        --light-gold: #FFF8DC;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Animated background pattern */
    @keyframes shimmer {
        0% { background-position: -100% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.4); }
        50% { box-shadow: 0 0 30px rgba(255, 215, 0, 0.8); }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Top Navigation Bar with enhanced styling */
    .top-nav {
        background: linear-gradient(135deg, #E70013 0%, #8B0000 50%, #E70013 100%);
        background-size: 200% 100%;
        animation: shimmer 6s ease-in-out infinite;
        padding: 1.5rem 2rem;
        border-radius: 0 0 25px 25px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(231, 0, 19, 0.4);
        border: 3px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .top-nav::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .nav-title {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.5), 0 0 20px rgba(255, 215, 0, 0.5);
        letter-spacing: 1.5px;
        position: relative;
        z-index: 1;
        animation: fadeIn 1s ease-in;
    }
    
    .nav-subtitle {
        color: #FFD700;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-style: italic;
        font-weight: 500;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Enhanced navigation tabs */
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    /* Main Header with dynamic effects */
    .main-header {
        background: linear-gradient(135deg, #E70013 0%, #C71585 50%, #8B0000 100%);
        background-size: 200% 200%;
        animation: shimmer 8s ease-in-out infinite;
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 40px rgba(231, 0, 19, 0.35);
        border: 4px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '🇹🇳';
        position: absolute;
        top: -20px;
        right: -20px;
        font-size: 10rem;
        opacity: 0.1;
        animation: pulse 4s ease-in-out infinite;
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 4px 4px 10px rgba(0,0,0,0.4), 0 0 30px rgba(255, 215, 0, 0.6);
        position: relative;
        z-index: 1;
        animation: slideIn 0.8s ease-out;
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin-top: 0.8rem;
        color: #FFD700;
        font-weight: 600;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        animation: slideIn 1s ease-out;
    }
    
    /* Enhanced metric cards with hover effects */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f9f9f9 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.12);
        border-left: 6px solid #E70013;
        border-top: 2px solid #FFD700;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: #1a1a1a;
        position: relative;
        overflow: hidden;
        animation: slideIn 0.6s ease-out;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card h3 {
        color: #E70013;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-card p {
        color: #333;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    .metric-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 15px 40px rgba(231, 0, 19, 0.25);
        border-left-color: #FFD700;
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Enhanced prediction result cards with animations */
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2.5rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.5);
        animation: slideIn 0.8s ease, pulse 3s ease-in-out infinite;
        border: 4px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-positive::before {
        content: '✨';
        position: absolute;
        font-size: 8rem;
        top: -30px;
        right: -30px;
        opacity: 0.2;
        animation: rotate 10s linear infinite;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2.5rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(235, 51, 73, 0.5);
        animation: slideIn 0.8s ease, pulse 3s ease-in-out infinite;
        border: 4px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-negative::before {
        content: '⚠️';
        position: absolute;
        font-size: 8rem;
        top: -30px;
        right: -30px;
        opacity: 0.2;
        animation: rotate 10s linear infinite;
    }
    
    /* Enhanced button styling */
    .stButton>button {
        background: linear-gradient(135deg, #E70013 0%, #C71585 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.9rem 2.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 6px 20px rgba(231, 0, 19, 0.4);
        border: 3px solid #FFD700;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 10px 35px rgba(231, 0, 19, 0.6);
    }
    
    /* Enhanced text area styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 4px solid #E70013;
        font-family: 'Arial', sans-serif;
        font-size: 1.15rem;
        padding: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(231, 0, 19, 0.2);
    }
    
    .stTextArea textarea:focus {
        border-color: #FFD700;
        box-shadow: 0 6px 25px rgba(255, 215, 0, 0.4);
        transform: scale(1.01);
    }
    
    /* Enhanced info box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border: 3px solid #FFD700;
        animation: slideIn 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: '💡';
        position: absolute;
        font-size: 12rem;
        bottom: -40px;
        right: -40px;
        opacity: 0.1;
    }
    
    .info-box h3 {
        color: #FFD700;
        margin-bottom: 1.2rem;
        font-size: 1.6rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .info-box ol {
        line-height: 2;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Enhanced feature cards */
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border-top: 5px solid #E70013;
        border-right: 2px solid #FFD700;
        margin-bottom: 1.5rem;
        color: #1a1a1a;
        transition: all 0.4s ease;
        animation: slideIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 0;
        background: linear-gradient(180deg, #E70013, #FFD700);
        transition: height 0.5s ease;
    }
    
    .feature-card:hover::before {
        height: 100%;
    }
    
    .feature-card h4 {
        color: #E70013;
        margin-bottom: 1rem;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .feature-card ul {
        color: #333;
        line-height: 2;
        font-size: 1.05rem;
    }
    
    .feature-card:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 35px rgba(231, 0, 19, 0.2);
        border-right-width: 4px;
    }
    
    /* Enhanced statistics section */
    .stats-header {
        background: linear-gradient(135deg, #E70013 0%, #8B0000 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        border: 3px solid #FFD700;
        box-shadow: 0 6px 25px rgba(231, 0, 19, 0.4);
        animation: slideIn 0.6s ease-out;
    }
    
    .stats-header h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
    }
    
    /* Tunisian flag accent with animation */
    .flag-accent {
        width: 100%;
        height: 6px;
        background: linear-gradient(90deg, 
            #E70013 0%, #E70013 45%, 
            #FFD700 45%, #FFD700 55%, 
            #E70013 55%, #E70013 100%);
        margin: 2.5rem 0;
        border-radius: 3px;
        box-shadow: 0 2px 10px rgba(231, 0, 19, 0.4);
        animation: shimmer 4s ease-in-out infinite;
        background-size: 200% 100%;
    }
    
    /* Enhanced metric display */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #E70013 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }
    
    /* Enhanced dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 2px solid #E70013;
    }
    
    /* Enhanced download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border: 3px solid #FFD700 !important;
        border-radius: 15px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(17, 153, 142, 0.4) !important;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 8px 30px rgba(17, 153, 142, 0.6) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #E70013, #FFD700, #E70013);
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 3px dashed #E70013;
        border-radius: 15px;
        padding: 2rem;
        background: linear-gradient(145deg, #fff 0%, #f9f9f9 100%);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #FFD700;
        background: linear-gradient(145deg, #FFF8DC 0%, #fff 100%);
        transform: scale(1.02);
    }
    
    /* Success/Error messages enhancement */
    .stAlert {
        border-radius: 15px;
        border-left: 6px solid #E70013;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #666;
        padding: 2rem 1rem;
        border-top: 3px solid #E70013;
        margin-top: 3rem;
        animation: fadeIn 1.5s ease-in;
    }
    
    .footer-text p {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Spinner enhancement */
    .stSpinner > div {
        border-top-color: #E70013 !important;
        border-right-color: #FFD700 !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Accueil"

st.markdown("""
<div class="top-nav">
    <h1 class="nav-title">🇹🇳 Analyse de Sentiment Tunisien - Élections</h1>
    <p class="nav-subtitle">✨ Plateforme MLOps pour l'analyse des opinions électorales ✨</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Accueil", use_container_width=True, type="primary" if st.session_state.current_page == "Accueil" else "secondary"):
        st.session_state.current_page = "Accueil"
        st.rerun()

with col2:
    if st.button("💬 Analyse", use_container_width=True, type="primary" if st.session_state.current_page == "Analyse" else "secondary"):
        st.session_state.current_page = "Analyse"
        st.rerun()

with col3:
    if st.button("📁 Analyse par Lot", use_container_width=True, type="primary" if st.session_state.current_page == "Lot" else "secondary"):
        st.session_state.current_page = "Lot"
        st.rerun()

with col4:
    if st.button("📈 Statistiques", use_container_width=True, type="primary" if st.session_state.current_page == "Stats" else "secondary"):
        st.session_state.current_page = "Stats"
        st.rerun()

st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)

if st.session_state.current_page == "Accueil":
    st.markdown("""
    <div class="main-header">
        <h1>🗳️ Analyse de Sentiment - Élections Tunisiennes</h1>
        <p>Comprendre l'opinion publique à travers l'intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Objectif</h3>
            <p>Analyser les sentiments des commentaires en dialecte tunisien relatifs aux élections pour comprendre l'opinion publique</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Technologie</h3>
            <p>Pipeline MLOps complet avec modèles d'apprentissage automatique adaptés au dialecte tunisien</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 Performance</h3>
            <p>Prédictions en temps réel avec haute précision et analyse instantanée</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🌟 Fonctionnalités Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>💬 Analyse Simple</h4>
            <ul>
                <li>Saisie de texte libre en dialecte tunisien</li>
                <li>Prédiction de sentiment instantanée (Pour/Contre)</li>
                <li>Score de confiance détaillé</li>
                <li>Visualisation interactive</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Statistiques</h4>
            <ul>
                <li>Historique des prédictions</li>
                <li>Distribution des sentiments</li>
                <li>Graphiques interactifs</li>
                <li>Analyse des tendances</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📁 Analyse par Lot</h4>
            <ul>
                <li>Upload de fichiers CSV</li>
                <li>Traitement de multiples textes</li>
                <li>Export des résultats</li>
                <li>Statistiques agrégées</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Précision</h4>
            <ul>
                <li>Modèle entraîné sur données tunisiennes</li>
                <li>Deux catégories : Pour (Positif) et Contre (Négatif)</li>
                <li>Score de confiance pour chaque prédiction</li>
                <li>Pipeline MLOps robuste</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>💡 Comment utiliser cette plateforme ?</h3>
        <ol>
            <li><strong>Analyse Simple</strong> : Cliquez sur "Analyse" pour analyser un texte en dialecte tunisien</li>
            <li><strong>Analyse par Lot</strong> : Téléchargez un fichier CSV pour analyser plusieurs commentaires</li>
            <li><strong>Statistiques</strong> : Consultez l'historique et les tendances de vos analyses</li>
            <li><strong>Résultats</strong> : Obtenez des prédictions Pour (1) ou Contre (0) avec un score de confiance</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "Analyse":
    st.markdown("### 💬 Analyse de Sentiment - Texte Simple")
    st.markdown("Entrez un commentaire en dialecte tunisien pour analyser son sentiment (Pour ou Contre)")
    
    text = st.text_area(
        "📝 Votre texte",
        value="",
        placeholder="مثال: هذا المرشح يعمل من أجل مصلحة الشعب التونسي",
        height=150,
        help="Entrez un commentaire en dialecte tunisien"
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        predict_button = st.button("🔍 Analyser", use_container_width=True)
    
    with col_btn2:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.rerun()
    
    if predict_button and text:
        with st.spinner("🔄 Analyse en cours..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            try:
                r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
                r.raise_for_status()
                result = r.json()
                
                st.success("✅ Analyse terminée avec succès!")
                
                # Robust extraction of prediction and confidence from varied API responses
                prediction_value = None
                confidence = None

                if isinstance(result, dict):
                    # possible keys for single prediction
                    if 'prediction' in result:
                        prediction_value = result['prediction']
                    elif 'sentiment' in result:
                        prediction_value = result['sentiment']
                    elif 'predictions' in result and isinstance(result['predictions'], (list, tuple)) and len(result['predictions']) > 0:
                        prediction_value = result['predictions'][0]
                    elif 'preds' in result and isinstance(result['preds'], (list, tuple)) and len(result['preds']) > 0:
                        prediction_value = result['preds'][0]
                    elif 'pred' in result:
                        prediction_value = result['pred']
                    elif 'label' in result:
                        prediction_value = result['label']

                    # extract confidence/score if present
                    for key in ('confidence', 'score', 'prob', 'probability', 'probabilities'):
                        if key in result:
                            confidence = result[key]
                            break
                else:
                    prediction_value = result

                # Normalize prediction_value to canonical string 'positive'/'negative'
                try:
                    # numeric labels
                    import numbers
                    if isinstance(prediction_value, str) and prediction_value.isdigit():
                        prediction_value = int(prediction_value)
                    if isinstance(prediction_value, numbers.Number):
                        prediction_value = 'positive' if int(prediction_value) == 1 else 'negative'
                    else:
                        prediction_value = str(prediction_value).lower()
                        if prediction_value in ('0', 'false', 'neg', 'negative', 'contre'):
                            prediction_value = 'negative'
                        elif prediction_value in ('1', 'true', 'pos', 'positive', 'pour'):
                            prediction_value = 'positive'
                except Exception:
                    prediction_value = str(prediction_value).lower()

                # Normalize confidence to float between 0 and 1
                try:
                    if isinstance(confidence, (list, tuple)) and len(confidence) > 0:
                        confidence = float(confidence[0])
                    else:
                        confidence = float(confidence) if confidence is not None else 0.0
                    # If confidence seems >1, assume it's percentage
                    if confidence > 1:
                        confidence = min(confidence / 100.0, 1.0)
                except Exception:
                    confidence = 0.0

                if prediction_value == 'positive':
                    sentiment_fr = 'Pour (Positif)'
                    sentiment_short = 'Pour'
                    sentiment_numeric = 1
                    is_positive = True
                else:
                    sentiment_fr = 'Contre (Négatif)'
                    sentiment_short = 'Contre'
                    sentiment_numeric = 0
                    is_positive = False
                
                st.session_state.prediction_history.append({
                    'text': text,
                    'sentiment': sentiment_short,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.markdown("---")
                st.markdown("<h3 style='text-align:center;'>🎯 Résultat de l'Analyse</h3>", unsafe_allow_html=True)

                cols = st.columns([1, 0.6, 1])

                with cols[1]:
                    if is_positive:
                        st.markdown(f"""
                        <div class="prediction-positive" style="text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;">
                            <div style="font-size:2rem;">✅</div>
                            <div style="font-weight:700;margin-top:0.5rem;">{sentiment_fr}</div>
                            <div style="font-size:1.05rem;margin-top:0.5rem;">Opinion favorable au candidat/programme</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-negative" style="text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;">
                            <div style="font-size:2rem;">❌</div>
                            <div style="font-weight:700;margin-top:0.5rem;">{sentiment_fr}</div>
                            <div style="font-size:1.05rem;margin-top:0.5rem;">Opinion défavorable au candidat/programme</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence and detailed breakdown removed per UX request
                
            except requests.exceptions.Timeout:
                st.error("⏱️ Délai d'attente dépassé. Veuillez réessayer.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Erreur de connexion à l'API. Vérifiez que le backend est en cours d'exécution.")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

elif st.session_state.current_page == "Lot":
    st.markdown("### 📁 Analyse par Lot - Fichier CSV")
    st.markdown("Téléchargez un fichier CSV contenant une colonne 'text' pour analyser plusieurs commentaires")
    
    uploaded = st.file_uploader(
        "📤 Choisir un fichier CSV",
        type=["csv"],
        help="Le fichier doit contenir une colonne nommée 'text'"
    )
    
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            
            if 'text' not in df.columns:
                st.warning("⚠️ Le fichier CSV doit contenir une colonne 'text'")
            else:
                st.markdown("---")
                st.markdown("### 🔄 Traitement en cours...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("⏳ Envoi du fichier au serveur...")
                
                try:
                    # Send as multipart/form-data with filename and content-type so FastAPI
                    # receives a proper UploadFile (with filename) instead of raw bytes.
                    files = {'file': (uploaded.name, uploaded.getvalue(), 'text/csv')}
                    r = requests.post(f"{API_URL}/predict_csv", files=files, timeout=60)
                    r.raise_for_status()
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Analyse terminée avec succès!")
                    
                    st.markdown("---")
                    st.markdown("### 📥 Télécharger les Résultats")
                    
                    st.download_button(
                        label="⬇️ Télécharger le CSV avec prédictions",
                        data=r.content,
                        file_name=f"predictions_{uploaded.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    results_df = pd.read_csv(pd.io.common.BytesIO(r.content))
                    st.markdown("#### 👀 Aperçu des Résultats")
                    st.dataframe(results_df.head(10), use_container_width=True)
                    
                    if 'sentiment' in results_df.columns or 'prediction' in results_df.columns:
                        sentiment_col = 'sentiment' if 'sentiment' in results_df.columns else 'prediction'
                        
                        results_df['sentiment_mapped'] = results_df[sentiment_col].apply(
                            lambda x: 'Pour' if str(x) in ['1', 'positive', 'positif', 'pour'] else 'Contre'
                        )
                        
                        st.markdown("### 📊 Distribution des Sentiments")
                        
                        sentiment_counts = results_df['sentiment_mapped'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="Répartition Pour/Contre",
                                color_discrete_sequence=['#38ef7d', '#f45c43']
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.bar(
                                x=sentiment_counts.index,
                                y=sentiment_counts.values,
                                title="Nombre par Sentiment",
                                labels={'x': 'Sentiment', 'y': 'Nombre'},
                                color=sentiment_counts.index,
                                color_discrete_sequence=['#38ef7d', '#f45c43']
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                except requests.exceptions.Timeout:
                    st.error("⏱️ Délai d'attente dépassé. Le fichier est peut-être trop volumineux.")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {str(e)}")

elif st.session_state.current_page == "Stats":
    st.markdown('<div class="stats-header"><h2>📈 Statistiques et Historique</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.prediction_history:
        st.info("📊 Aucune prédiction enregistrée pour le moment. Effectuez des analyses pour voir les statistiques.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Prédictions", len(history_df))
        
        with col2:
            pour_count = len(history_df[history_df['sentiment'] == 'Pour'])
            st.metric("✅ Pour (Positif)", pour_count)
        
        with col3:
            contre_count = len(history_df[history_df['sentiment'] == 'Contre'])
            st.metric("❌ Contre (Négatif)", contre_count)
        
        st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🥧 Distribution des Sentiments")
            sentiment_counts = history_df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color_discrete_sequence=['#38ef7d', '#f45c43']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Confiance Moyenne par Sentiment")
            avg_confidence = history_df.groupby('sentiment')['confidence'].mean() * 100
            fig_bar = px.bar(
                x=avg_confidence.index,
                y=avg_confidence.values,
                title="",
                labels={'x': 'Sentiment', 'y': 'Confiance (%)'},
                color=avg_confidence.index,
                color_discrete_sequence=['#38ef7d', '#f45c43']
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
        
        st.markdown("#### 📜 Historique des Prédictions")
        
        display_df = history_df.copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(
            display_df[['timestamp', 'text', 'sentiment', 'confidence']].sort_values('timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("🗑️ Effacer l'Historique", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()

st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
    <p style="font-size: 1.2rem; font-weight: 700;">🇹🇳 Analyse de Sentiment Tunisien | MLOps Project © 2025</p>
    <p style="font-size: 1rem; color: #E70013; font-weight: 600;">✨ Développé pour les élections tunisiennes avec ❤️ ✨</p>
</div>
""", unsafe_allow_html=True)
