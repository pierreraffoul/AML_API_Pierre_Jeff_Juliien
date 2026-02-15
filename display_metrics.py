#!/usr/bin/env python3
"""
Script pour afficher les métriques des modèles SVM et Random Forest.
Ce script est indépendant de l'API et peut être exécuté directement.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
import os
from dotenv import load_dotenv
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Charger les variables d'environnement
load_dotenv()

# Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://lqckcteuponqeisgovhr.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_-2Fofz2Gxy2hqabBgt5w0A_b7k2eihI")

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def get_data_from_supabase() -> pd.DataFrame:
    """Récupère les données depuis Supabase."""
    print("📡 Récupération des données depuis Supabase...")
    
    url = SUPABASE_URL.rstrip('/')
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    
    with httpx.Client() as client:
        response = client.get(
            f"{url}/rest/v1/ai_training_data",
            params={"select": "*"},
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        df = pd.DataFrame(response.json())
    
    print(f"✅ {len(df)} lignes récupérées")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données."""
    print("🧹 Nettoyage des données...")
    
    # Conversion date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Colonnes numériques critiques
    numeric_cols = [
        'cote_dom_clean', 'cote_nul_clean', 'cote_ext_clean',
        'home_forme_pts_last5', 'home_moy_buts_marques_last5', 'home_moy_buts_encaisse_last5',
        'away_forme_pts_last5', 'away_moy_buts_marques_last5', 'away_moy_buts_encaisse_last5'
    ]
    
    # Force la conversion en numérique
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Supprime les lignes avec des NaN
    initial_len = len(df)
    df = df.dropna(subset=numeric_cols + ['ftr'])
    removed = initial_len - len(df)
    
    print(f"   📉 {removed} lignes supprimées (données manquantes)")
    print(f"   ✅ {len(df)} lignes restantes")
    
    return df


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Génère et affiche la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Réalité', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   🖼️  Graphique sauvegardé: {filename}")
    plt.close()


def print_metrics(y_true, y_pred, y_proba, labels, model_name):
    """Affiche les métriques détaillées d'un modèle."""
    print(f"\n{'='*60}")
    print(f"📊 MÉTRIQUES - {model_name}")
    print(f"{'='*60}")
    
    # Métriques globales
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Accuracy (Précision globale): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Métriques par classe
    precision = precision_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
    
    print(f"\n📈 Métriques par classe:")
    print(f"{'Classe':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for i, label in enumerate(labels):
        label_name = {"H": "Domicile 🏠", "D": "Nul 🤝", "A": "Extérieur ✈️"}.get(label, label)
        print(f"{label_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    
    # Métriques moyennes
    print(f"\n📊 Moyennes:")
    print(f"   Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"   Recall (macro):    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"   F1-Score (macro):  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    
    # AUC Score si disponible
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            print(f"   AUC-ROC (ovr):     {auc:.4f}")
        except Exception as e:
            print(f"   AUC-ROC:           Non calculable ({str(e)[:50]})")
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Matrice de confusion:")
    print(f"   {'':<10}", end="")
    for label in labels:
        label_name = {"H": "Dom", "D": "Nul", "A": "Ext"}.get(label, label)
        print(f"{label_name:>8}", end="")
    print()
    
    for i, label in enumerate(labels):
        label_name = {"H": "Dom", "D": "Nul", "A": "Ext"}.get(label, label)
        print(f"   {label_name:<10}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>8}", end="")
        print()
    
    # Classification report
    print(f"\n📋 Rapport de classification détaillé:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


# ==========================================
# ENTRÂINEMENT ET ÉVALUATION
# ==========================================

def train_and_evaluate_models(df: pd.DataFrame):
    """Entraîne et évalue les modèles SVM et Random Forest."""
    print("\n" + "="*60)
    print("🤖 ENTRAÎNEMENT DES MODÈLES")
    print("="*60)
    
    # Features
    features = [
        'cote_dom_clean', 'cote_nul_clean', 'cote_ext_clean',
        'home_forme_pts_last5', 'away_forme_pts_last5',
        'home_moy_buts_marques_last5', 'away_moy_buts_encaisse_last5'
    ]
    
    X = df[features]
    y = df['ftr']
    
    # Encodage de la cible
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = le.classes_
    
    print(f"\n📊 Données:")
    print(f"   Features: {len(features)}")
    print(f"   Échantillons: {len(X)}")
    print(f"   Classes: {labels}")
    print(f"   Distribution: {dict(zip(labels, np.bincount(y_encoded)))}")
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"\n📦 Split des données:")
    print(f"   Train: {len(X_train)} échantillons")
    print(f"   Test:  {len(X_test)} échantillons")
    
    # Scaling pour SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==========================================
    # RANDOM FOREST
    # ==========================================
    print(f"\n{'='*60}")
    print("🌲 RANDOM FOREST")
    print(f"{'='*60}")
    
    print("\n🔍 Recherche des meilleurs hyperparamètres (GridSearch)...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=3,
        scoring='roc_auc_ovr',
        n_jobs=-1,
        verbose=1
    )
    grid_search_rf.fit(X_train, y_train)
    
    print(f"✅ Meilleurs paramètres trouvés:")
    for param, value in grid_search_rf.best_params_.items():
        print(f"   {param}: {value}")
    
    # Entraînement avec paramètres optimisés
    print("\n🌲 Entraînement Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)
    
    # Affichage des métriques
    print_metrics(y_test, y_pred_rf, y_proba_rf, labels, "Random Forest")
    plot_confusion_matrix(y_test, y_pred_rf, labels, 
                         "Matrice de Confusion - Random Forest", 
                         "confusion_matrix_rf.png")
    
    # ==========================================
    # SVM
    # ==========================================
    print(f"\n{'='*60}")
    print("🛡️  SVM (Support Vector Machine)")
    print(f"{'='*60}")
    
    print("\n🛡️  Entraînement SVM...")
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_proba_svm = svm_model.predict_proba(X_test_scaled)
    
    # Affichage des métriques
    print_metrics(y_test, y_pred_svm, y_proba_svm, labels, "SVM")
    plot_confusion_matrix(y_test, y_pred_svm, labels,
                          "Matrice de Confusion - SVM",
                          "confusion_matrix_svm.png")
    
    # ==========================================
    # COMPARAISON
    # ==========================================
    print(f"\n{'='*60}")
    print("⚖️  COMPARAISON DES MODÈLES")
    print(f"{'='*60}")
    
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    
    print(f"\n{'Modèle':<20} {'Accuracy':<12} {'F1-Score (macro)':<18}")
    print("-" * 60)
    print(f"{'Random Forest':<20} {rf_accuracy:<12.4f} {f1_score(y_test, y_pred_rf, average='macro', zero_division=0):<18.4f}")
    print(f"{'SVM':<20} {svm_accuracy:<12.4f} {f1_score(y_test, y_pred_svm, average='macro', zero_division=0):<18.4f}")
    
    meilleur = "Random Forest" if rf_accuracy > svm_accuracy else "SVM"
    print(f"\n🏆 Meilleur modèle: {meilleur}")
    
    print(f"\n✅ Analyse terminée!")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("📊 AFFICHAGE DES MÉTRIQUES - SVM & RANDOM FOREST")
    print("="*60)
    
    try:
        # 1. Récupération des données
        df = get_data_from_supabase()
        
        # 2. Nettoyage
        df_clean = clean_data(df)
        
        # 3. Entraînement et évaluation
        train_and_evaluate_models(df_clean)
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

