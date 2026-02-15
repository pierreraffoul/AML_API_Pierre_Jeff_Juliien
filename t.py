import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client, Client
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
from dotenv import load_dotenv

# --- CONFIGURATION SUPABASE ---
# Charger les variables d'environnement depuis .env
load_dotenv()

URL = os.getenv("SUPABASE_URL", "https://lqckcteuponqeisgovhr.supabase.co")
KEY = os.getenv("SUPABASE_KEY", "sb_publishable_-2Fofz2Gxy2hqabBgt5w0A_b7k2eihI")
supabase: Client = create_client(URL, KEY)

# ==========================================
# 1. RÉCUPÉRATION ET NETTOYAGE
# ==========================================
def get_and_clean_data():
    print("📡 Récupération des données depuis Supabase...")
    response = supabase.table("ai_training_data").select("*").execute()
    df = pd.DataFrame(response.data)
    
    # Nettoyage
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    numeric_cols = [
        'cote_dom_clean', 'cote_nul_clean', 'cote_ext_clean',
        'home_forme_pts_last5', 'home_moy_buts_marques_last5', 'home_moy_buts_encaisse_last5',
        'away_forme_pts_last5', 'away_moy_buts_marques_last5', 'away_moy_buts_encaisse_last5'
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=numeric_cols + ['ftr'])
    print(f"✅ Données prêtes pour analyse : {len(df)} matchs.")
    return df

# ==========================================
# 2. ANALYSE D'IMPORTANCE
# ==========================================
def analyze_importance(df):
    print("\n🕵️‍♀️ Calcul de l'importance des variables en cours...")
    
    # Liste EXACTE des colonnes utilisées pour l'entraînement
    features_cols = [
        'cote_dom_clean', 
        'cote_nul_clean', 
        'cote_ext_clean',
        'home_forme_pts_last5', 
        'away_forme_pts_last5',
        'home_moy_buts_marques_last5', 
        'away_moy_buts_encaisse_last5'
    ]
    
    # Noms plus lisibles pour le graphique
    feature_names_clean = [
        'Cote Domicile', 
        'Cote Nul', 
        'Cote Extérieur',
        'Forme Dom (5 derniers)', 
        'Forme Ext (5 derniers)',
        'Attaque Domicile', 
        'Défense Extérieur'
    ]
    
    X = df[features_cols]
    y = df['ftr']
    
    # Encodage de la cible
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # On utilise le Random Forest avec les paramètres optimisés trouvés précédemment
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=5, 
        random_state=42
    )
    
    rf_model.fit(X, y_encoded)
    
    # --- EXTRACTION DES IMPORTANCES ---
    importances = rf_model.feature_importances_
    
    # Création d'un tableau pour trier les résultats
    feature_imp_df = pd.DataFrame({
        'Variable': feature_names_clean,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Affichage console
    print("\n🏆 CLASSEMENT : QU'EST-CE QUI COMPTE POUR L'IA ? (Total = 1.0)")
    print(feature_imp_df)
    
    # --- VISUALISATION ---
    plt.figure(figsize=(10, 6))
    
    # Graphique en barres
    sns.barplot(
        x='Importance', 
        y='Variable', 
        data=feature_imp_df, 
        palette='viridis',
        hue='Variable',
        legend=False
    )
    
    plt.title('Importance des Features (Poids dans la décision)', fontsize=15)
    plt.xlabel('Importance (0 à 1)', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Sauvegarde
    filename = "feature_importance_analysis.png"
    plt.savefig(filename)
    plt.close()
    print(f"\n🖼️ Graphique sauvegardé sous : {filename}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # 1. Récupération
    df = get_and_clean_data()
    
    # 2. Analyse
    analyze_importance(df)