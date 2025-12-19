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
    /* Tunisian-inspired sophisticated color palette */
    :root {
        --primary-burgundy: #8B1538;
        --soft-terracotta: #D4755B;
        --warm-sand: #F5E6D3;
        --deep-navy: #2C3E50;
        --golden-accent: #D4AF37;
        --soft-blush: #F8EDE3;
        --olive-green: #6B7F5C;
        --deep-crimson: #9B1B30;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Smooth animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes floatPattern {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    @keyframes shimmerGold {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(212, 175, 55, 0.3); }
        50% { box-shadow: 0 0 40px rgba(212, 175, 55, 0.6); }
    }
    
    @keyframes slideFromRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Global styling with Tunisian aesthetic */
    .stApp {
        background: linear-gradient(135deg, #F8EDE3 0%, #F5E6D3 50%, #FFF 100%);
        font-family: 'Georgia', 'Palatino', serif;
    }
    
    /* Elegant top navigation with Tunisian patterns */
    .top-nav {
        background: linear-gradient(135deg, #8B1538 0%, #9B1B30 50%, #8B1538 100%);
        padding: 2.5rem 3rem;
        border-radius: 0;
        margin-bottom: 0;
        box-shadow: 0 6px 25px rgba(139, 21, 56, 0.25);
        position: relative;
        overflow: hidden;
        border-bottom: 4px solid #D4AF37;
    }
    
    .top-nav::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(212, 175, 55, 0.03) 35px, rgba(212, 175, 55, 0.03) 70px);
        pointer-events: none;
    }
    
    .nav-title {
        color: #F5E6D3;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .nav-subtitle {
        color: #D4AF37;
        text-align: center;
        font-size: 1rem;
        margin-top: 0.75rem;
        font-style: italic;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 1px;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Refined navigation tabs */
    .stButton>button {
        background: transparent;
        color: #2C3E50;
        border: 2px solid #8B1538;
        border-radius: 8px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 8px rgba(139, 21, 56, 0.15);
        background: linear-gradient(135deg, #FFF 0%, #F8EDE3 100%);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #8B1538 0%, #9B1B30 100%);
        color: #F5E6D3;
        border-color: #D4AF37;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(139, 21, 56, 0.3);
    }
    
    /* Active button state */
    .stButton>button[data-baseweb="button"] {
        background: linear-gradient(135deg, #8B1538 0%, #9B1B30 100%);
        color: #F5E6D3;
        border-color: #D4AF37;
    }
    
    /* Main header with elegant design */
    .main-header {
        background: linear-gradient(135deg, #8B1538 0%, #6B7F5C 100%);
        padding: 3.5rem 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 3rem 0 2.5rem;
        box-shadow: 0 8px 30px rgba(139, 21, 56, 0.2);
        border: 3px solid #D4AF37;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .main-header::after {
        content: '🗳️';
        position: absolute;
        top: -30px;
        right: -30px;
        font-size: 12rem;
        opacity: 0.08;
        animation: floatPattern 6s ease-in-out infinite;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        color: #F5E6D3;
        letter-spacing: 0.5px;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 1rem;
        color: #D4AF37;
        font-weight: 400;
        position: relative;
        z-index: 1;
        font-style: italic;
    }
    
    /* Sophisticated metric cards */
    .metric-card {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8EDE3 100%);
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #8B1538;
        border-bottom: 2px solid #D4AF37;
        transition: all 0.4s ease;
        color: #2C3E50;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
        margin-bottom: 1.5rem;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -2px;
        right: -2px;
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), transparent);
        border-radius: 0 12px 0 0;
    }
    
    .metric-card h3 {
        color: #8B1538;
        margin-bottom: 1rem;
        font-size: 1.4rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    
    .metric-card p {
        color: #2C3E50;
        line-height: 1.7;
        font-size: 1rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 35px rgba(139, 21, 56, 0.15);
        border-left-width: 8px;
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    /* Enhanced prediction cards with refined styling */
    .prediction-positive {
        background: linear-gradient(135deg, #6B7F5C 0%, #8BA888 100%);
        padding: 3rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(107, 127, 92, 0.3);
        animation: fadeInUp 0.8s ease;
        border: 3px solid #D4AF37;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-positive::before {
        content: '✓';
        position: absolute;
        font-size: 10rem;
        top: -40px;
        right: -40px;
        opacity: 0.1;
        font-weight: 700;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #8B1538 0%, #9B1B30 100%);
        padding: 3rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(139, 21, 56, 0.3);
        animation: fadeInUp 0.8s ease;
        border: 3px solid #D4AF37;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-negative::before {
        content: '✗';
        position: absolute;
        font-size: 10rem;
        top: -40px;
        right: -40px;
        opacity: 0.1;
        font-weight: 700;
    }
    
    /* Refined text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 3px solid #8B1538;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 1.05rem;
        padding: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 3px 12px rgba(139, 21, 56, 0.15);
        background: #FFFFFF;
    }
    
    .stTextArea textarea:focus {
        border-color: #D4AF37;
        box-shadow: 0 5px 20px rgba(212, 175, 55, 0.25);
        outline: none;
    }
    
    /* Sophisticated info box */
    .info-box {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin: 2.5rem 0;
        box-shadow: 0 6px 20px rgba(44, 62, 80, 0.25);
        border: 2px solid #D4AF37;
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 40px,
            rgba(212, 175, 55, 0.03) 40px,
            rgba(212, 175, 55, 0.03) 80px
        );
        pointer-events: none;
    }
    
    .info-box h3 {
        color: #D4AF37;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .info-box ol {
        line-height: 2;
        font-size: 1.05rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Refined feature cards */
    .feature-card {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8EDE3 100%);
        padding: 2.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border-top: 4px solid #8B1538;
        border-right: 2px solid #D4AF37;
        margin-bottom: 1.5rem;
        color: #2C3E50;
        transition: all 0.4s ease;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
    }
    
    .feature-card h4 {
        color: #8B1538;
        margin-bottom: 1.2rem;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .feature-card ul {
        color: #2C3E50;
        line-height: 1.9;
        font-size: 1rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .feature-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 30px rgba(139, 21, 56, 0.15);
        border-right-width: 4px;
    }
    
    /* Elegant statistics section */
    .stats-header {
        background: linear-gradient(135deg, #8B1538 0%, #6B7F5C 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2.5rem;
        border: 3px solid #D4AF37;
        box-shadow: 0 5px 20px rgba(139, 21, 56, 0.25);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .stats-header h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #F5E6D3;
    }
    
    /* Tunisian decorative accent */
    .flag-accent {
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, 
            #8B1538 0%, #8B1538 48%, 
            #D4AF37 48%, #D4AF37 52%, 
            #8B1538 52%, #8B1538 100%);
        margin: 2.5rem 0;
        border-radius: 3px;
        box-shadow: 0 2px 8px rgba(139, 21, 56, 0.2);
    }
    
    /* Refined metric display */
    [data-testid="stMetricValue"] {
        font-size: 2.3rem !important;
        font-weight: 700 !important;
        color: #8B1538 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #2C3E50 !important;
    }
    
    /* Enhanced dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border: 2px solid #8B1538;
    }
    
    /* Refined download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #6B7F5C 0%, #8BA888 100%) !important;
        color: white !important;
        border: 2px solid #D4AF37 !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(107, 127, 92, 0.25) !important;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(107, 127, 92, 0.35) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8B1538, #D4AF37, #8B1538);
        background-size: 200% 100%;
        animation: shimmerGold 2s linear infinite;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 3px dashed #8B1538;
        border-radius: 12px;
        padding: 2rem;
        background: linear-gradient(145deg, #FFF 0%, #F8EDE3 100%);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #D4AF37;
        background: linear-gradient(145deg, #F8EDE3 0%, #F5E6D3 100%);
    }
    
    /* Alert messages enhancement */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #8B1538;
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #2C3E50;
        padding: 2.5rem 1rem;
        border-top: 2px solid #D4AF37;
        margin-top: 3rem;
        animation: fadeInUp 1.5s ease-in;
        background: linear-gradient(135deg, #F8EDE3 0%, #FFF 100%);
    }
    
    .footer-text p {
        font-size: 1rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    /* Spinner enhancement */
    .stSpinner > div {
        border-top-color: #8B1538 !important;
        border-right-color: #D4AF37 !important;
    }
    
    /* Cultural pattern overlay */
    .cultural-pattern {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.02;
        background-image: 
            repeating-linear-gradient(45deg, transparent, transparent 50px, #8B1538 50px, #8B1538 51px),
            repeating-linear-gradient(-45deg, transparent, transparent 50px, #D4AF37 50px, #D4AF37 51px);
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
    <h1 class="nav-title">🇹🇳 Analyse de Sentiment - Élections Tunisiennes</h1>
    <p class="nav-subtitle">Plateforme d'Intelligence Artificielle pour la Démocratie</p>
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
            <li><strong>Résultats</strong> : Obtenez des prédictions précises avec scores de confiance</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="footer-text"><p>🇹🇳 Plateforme développée pour analyser l\'opinion publique tunisienne</p></div>', unsafe_allow_html=True)

elif st.session_state.current_page == "Analyse":
    st.markdown("""
    <div class="main-header">
        <h1>💬 Analyse de Sentiment Individuelle</h1>
        <p>Analysez un commentaire en dialecte tunisien</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Entrez le texte à analyser:",
        height=150,
        placeholder="Exemple: المرشح هذا باهي برشا، نحبو ونتمنى يربح"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analyser le Sentiment", use_container_width=True)
    
    if analyze_button:
        if user_input.strip():
            with st.spinner("🔄 Analyse en cours..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"text": user_input},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        sentiment = result.get("sentiment", "Unknown")
                        confidence = result.get("confidence", 0)
                        
                        st.session_state.prediction_history.append({
                            "text": user_input,
                            "sentiment": sentiment,
                            "confidence": confidence,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
                        
                        if sentiment == "Pour":
                            st.markdown(f"""
                            <div class="prediction-positive">
                                ✅ Sentiment: <strong>{sentiment}</strong><br>
                                Score de confiance: <strong>{confidence:.2%}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-negative">
                                ❌ Sentiment: <strong>{sentiment}</strong><br>
                                Score de confiance: <strong>{confidence:.2%}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                label="Sentiment Détecté",
                                value=sentiment,
                                delta="Positif" if sentiment == "Pour" else "Négatif"
                            )
                        
                        with col2:
                            st.metric(
                                label="Confiance",
                                value=f"{confidence:.2%}",
                                delta=f"{confidence:.2%}"
                            )
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=confidence * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Niveau de Confiance", 'font': {'size': 24, 'color': '#2C3E50'}},
                            delta={'reference': 50, 'increasing': {'color': "#6B7F5C"}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#2C3E50"},
                                'bar': {'color': "#8B1538"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "#2C3E50",
                                'steps': [
                                    {'range': [0, 50], 'color': '#F5E6D3'},
                                    {'range': [50, 75], 'color': '#D4AF37'},
                                    {'range': [75, 100], 'color': '#6B7F5C'}
                                ],
                                'threshold': {
                                    'line': {'color': "#8B1538", 'width': 4},
                                    'thickness': 0.75,
                                    'value': confidence * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            paper_bgcolor="#F8EDE3",
                            font={'color': "#2C3E50", 'family': "Arial"}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success("✅ Analyse terminée avec succès!")
                    else:
                        st.error(f"❌ Erreur API: {response.status_code} - {response.text}")
                
                except requests.exceptions.Timeout:
                    st.error("⏱️ La requête a expiré. Veuillez réessayer.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Impossible de se connecter au serveur API.")
                except Exception as e:
                    st.error(f"❌ Une erreur est survenue: {str(e)}")
        else:
            st.warning("⚠️ Veuillez entrer un texte avant d'analyser.")

elif st.session_state.current_page == "Lot":
    st.markdown("""
    <div class="main-header">
        <h1>📁 Analyse par Lot</h1>
        <p>Analysez plusieurs commentaires simultanément</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>📋 Format du fichier CSV</h3>
        <p>Votre fichier doit contenir une colonne nommée <strong>'text'</strong> avec les commentaires à analyser.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Téléchargez votre fichier CSV",
        type=['csv'],
        help="Le fichier doit contenir une colonne 'text'"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("❌ Le fichier doit contenir une colonne 'text'")
            else:
                st.success(f"✅ Fichier chargé: {len(df)} lignes détectées")
                
                st.markdown("### 📊 Aperçu des données")
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if st.button("🚀 Lancer l'Analyse par Lot", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        total = len(df)
                        
                        for idx, row in df.iterrows():
                            status_text.text(f"Analyse en cours: {idx + 1}/{total}")
                            progress_bar.progress((idx + 1) / total)
                            
                            try:
                                response = requests.post(
                                    f"{API_URL}/predict",
                                    json={"text": str(row['text'])},
                                    timeout=10
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    results.append({
                                        'text': row['text'],
                                        'sentiment': result.get('sentiment', 'Unknown'),
                                        'confidence': result.get('confidence', 0)
                                    })
                                else:
                                    results.append({
                                        'text': row['text'],
                                        'sentiment': 'Error',
                                        'confidence': 0
                                    })
                            
                            except Exception as e:
                                results.append({
                                    'text': row['text'],
                                    'sentiment': 'Error',
                                    'confidence': 0
                                })
                            
                            time.sleep(0.1)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        results_df = pd.DataFrame(results)
                        
                        st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
                        st.markdown("### 📈 Résultats de l'Analyse")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pour_count = (results_df['sentiment'] == 'Pour').sum()
                            st.metric("Pour", pour_count, f"{pour_count/len(results_df)*100:.1f}%")
                        
                        with col2:
                            contre_count = (results_df['sentiment'] == 'Contre').sum()
                            st.metric("Contre", contre_count, f"{contre_count/len(results_df)*100:.1f}%")
                        
                        with col3:
                            avg_conf = results_df['confidence'].mean()
                            st.metric("Confiance Moyenne", f"{avg_conf:.2%}")
                        
                        sentiment_counts = results_df['sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Distribution des Sentiments",
                            color_discrete_map={'Pour': '#6B7F5C', 'Contre': '#8B1538', 'Error': '#D4755B'}
                        )
                        fig.update_layout(
                            paper_bgcolor='#F8EDE3',
                            plot_bgcolor='#F8EDE3',
                            font={'color': '#2C3E50', 'size': 14}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("### 📋 Résultats Détaillés")
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Télécharger les Résultats (CSV)",
                            data=csv,
                            file_name=f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")

elif st.session_state.current_page == "Stats":
    st.markdown("""
    <div class="stats-header">
        <h2>📊 Statistiques et Historique</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.prediction_history:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total d'Analyses", len(df_history))
        
        with col2:
            pour_count = (df_history['sentiment'] == 'Pour').sum()
            st.metric("Pour", pour_count, f"{pour_count/len(df_history)*100:.1f}%")
        
        with col3:
            contre_count = (df_history['sentiment'] == 'Contre').sum()
            st.metric("Contre", contre_count, f"{contre_count/len(df_history)*100:.1f}%")
        
        with col4:
            avg_confidence = df_history['confidence'].mean()
            st.metric("Confiance Moyenne", f"{avg_confidence:.2%}")
        
        st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df_history['sentiment'].value_counts()
            fig1 = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Distribution des Sentiments",
                labels={'x': 'Sentiment', 'y': 'Nombre'},
                color=sentiment_counts.index,
                color_discrete_map={'Pour': '#6B7F5C', 'Contre': '#8B1538'}
            )
            fig1.update_layout(
                paper_bgcolor='#F8EDE3',
                plot_bgcolor='#FFF',
                font={'color': '#2C3E50', 'size': 12}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(
                df_history,
                x='confidence',
                nbins=20,
                title="Distribution de la Confiance",
                labels={'confidence': 'Confiance', 'count': 'Nombre'},
                color_discrete_sequence=['#8B1538']
            )
            fig2.update_layout(
                paper_bgcolor='#F8EDE3',
                plot_bgcolor='#FFF',
                font={'color': '#2C3E50', 'size': 12}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### 📜 Historique des Prédictions")
        st.dataframe(
            df_history[['timestamp', 'text', 'sentiment', 'confidence']].sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter l'Historique (CSV)",
            data=csv,
            file_name=f"historique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Effacer l'Historique", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Aucune donnée disponible</h3>
            <p>Commencez par analyser des textes pour voir les statistiques apparaître ici.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="footer-text"><p>🇹🇳 Analyse de Sentiment Tunisien - Plateforme MLOps</p></div>', unsafe_allow_html=True)

st.markdown('<div class="flag-accent"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
    <p style="font-size: 1.2rem; font-weight: 700;">🇹🇳 Analyse de Sentiment Tunisien | MLOps Project © 2025</p>
    <p style="font-size: 1rem; color: #E70013; font-weight: 600;">✨ Développé pour les élections tunisiennes avec ❤️ ✨</p>
</div>
""", unsafe_allow_html=True)
