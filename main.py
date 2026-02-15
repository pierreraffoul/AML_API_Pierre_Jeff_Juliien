import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

# --- CONFIGURATION SUPABASE ---
# Charger les variables d'environnement depuis .env
load_dotenv()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(URL, KEY)

# ==========================================
# UTILITAIRES
# ==========================================

def get_data():
    """Récupère les données de Supabase."""
    print("📡 Récupération des données depuis Supabase...")
    response = supabase.table("ai_training_data").select("*").execute()
    df = pd.DataFrame(response.data)
    return df

def clean_data(df):
    """Nettoie les données (erreurs de types, dates dans colonnes numériques)."""
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
    print(f"   📉 Lignes supprimées (erreurs/dates) : {initial_len - len(df)}")
    print(f"   ✅ Lignes restantes : {len(df)}")
    
    return df

def traduire_resultat(code):
    """Traduit H/D/A en texte lisible."""
    if code == 'H': return "Victoire Domicile 🏠"
    if code == 'A': return "Victoire Extérieur ✈️"
    return "Match Nul 🤝"

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"🖼️ Matrice sauvegardée sous : {filename}")

# ==========================================
# 1. CLASSIFICATION (Random Forest & SVM)
# ==========================================

# N'oublie pas d'importer GridSearchCV si tu veux aller plus loin, 
# mais ici on fait un réglage manuel optimisé.

def run_classification(df):
    print("\n🤖 --- DÉBUT CLASSIFICATION OPTIMISÉE ---")
    
    features = [
        'cote_dom_clean', 'cote_nul_clean', 'cote_ext_clean',
        'home_forme_pts_last5', 'away_forme_pts_last5',
        'home_moy_buts_marques_last5', 'away_moy_buts_encaisse_last5'
    ]
    
    X = df[features]
    y = df['ftr']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = le.classes_
    
    # On garde 20% pour le test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🔍 Recherche des meilleurs hyperparamètres (GridSearch)...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc_ovr')
    grid_search.fit(X_train, y_train)
    
    print(f"🏆 Meilleurs paramètres trouvés : {grid_search.best_params_}")
        
    rf_model = grid_search.best_estimator_
    # Le reste (predict, proba...) reste identique
    
    # --- AMÉLIORATION 1 : RANDOM FOREST TUNÉ ---
    print("\n🌲 Entraînement Random Forest (Optimisé)...")
    
    # class_weight='balanced' : Force le modèle à prêter attention aux Nuls
    # max_depth=10 : Empêche le "par cœur" (overfitting)
    # n_estimators=200 : Plus d'arbres pour plus de stabilité
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_leaf=4,
        class_weight='balanced', 
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    proba_rf = rf_model.predict_proba(X_test)
    
    print("📊 Résultats Random Forest :")
    print(classification_report(y_test, y_pred_rf, target_names=labels))
    plot_confusion_matrix(y_test, y_pred_rf, labels, "Matrice RF Optimisée", "confusion_matrix_rf.png")
    
    # --- AMÉLIORATION 2 : SVM TUNÉ ---
    print("\n🛡️ Entraînement SVM (Optimisé)...")
    # C=1.0 et kernel rbf sont standard, mais class_weight aide aussi ici
    svm_model = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    
    print("📊 Résultats SVM :")
    print(classification_report(y_test, y_pred_svm, target_names=labels))
    plot_confusion_matrix(y_test, y_pred_svm, labels, "Matrice SVM Optimisée", "confusion_matrix_svm.png")

    print("✅ Modèles entraînés.")

    try:
        auc_score = roc_auc_score(y_test, proba_rf, multi_class='ovr') 
        print(f"🌟 Score AUC Global (Random Forest) : {auc_score:.4f}")
    except Exception as e:
        print(f"⚠️ Erreur calcul AUC: {e}")

    return rf_model, svm_model, scaler, le


# ==========================================
# 2. RÉGRESSION LINÉAIRE (Évolution des cotes)
# ==========================================

def run_regression(df, team_name):
    print(f"\n📈 --- DÉBUT RÉGRESSION LINÉAIRE ({team_name}) ---")
    
    # Filtrer l'équipe
    team_df = df[(df['hometeam'] == team_name) | (df['awayteam'] == team_name)].copy()
    
    if len(team_df) < 10:
        print(f"❌ Pas assez de données pour {team_name}. Essaie une autre équipe.")
        return

    # Préparer la cote à analyser
    team_df['ma_cote'] = np.where(team_df['hometeam'] == team_name, 
                                  team_df['cote_dom_clean'], 
                                  team_df['cote_ext_clean'])
    
    team_df = team_df.sort_values('date')
    team_df['time_index'] = np.arange(len(team_df))
    
    X = team_df[['time_index']]
    y = team_df['ma_cote']
    
    # Entraînement
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    # Analyse
    coef = reg.coef_[0]
    tendance = "en hausse ↗️" if coef > 0 else "en baisse ↘️"
    msg_confiance = "(L'équipe est moins favorite)" if coef > 0 else "(L'équipe est plus favorite)"
    
    print(f"Coefficient (Pente) : {coef:.4f}")
    print(f"👉 Les cotes de {team_name} sont globalement {tendance} {msg_confiance}.")

    # Sauvegarde du Graphique
    plt.figure(figsize=(10, 6))
    plt.scatter(team_df['date'], y, color='blue', alpha=0.4, label='Cotes réelles')
    plt.plot(team_df['date'], y_pred, color='red', linewidth=2, label='Tendance')
    plt.title(f"Évolution des cotes de victoire : {team_name}")
    plt.xlabel("Années")
    plt.ylabel("Cote")
    plt.legend()
    
    filename = f"regression_{team_name}.png"
    plt.savefig(filename)
    plt.close()
    print(f"🖼️ Graphique sauvegardé sous : {filename}")

# ==========================================
# 3. PRÉDICTION (INFERENCE)
# ==========================================

def predire_un_match(rf_model, svm_model, scaler, le, match_data):
    print(f"\n🔮 --- PRÉDICTION : {match_data['hometeam']} vs {match_data['awayteam']} ---")

    # Colonnes EXACTEMENT comme à l'entraînement
    features_names = [
        'cote_dom_clean', 'cote_nul_clean', 'cote_ext_clean',
        'home_forme_pts_last5', 'away_forme_pts_last5',
        'home_moy_buts_marques_last5', 'away_moy_buts_encaisse_last5'
    ]
    
    # Création DataFrame
    input_data = pd.DataFrame([match_data], columns=['hometeam', 'awayteam'] + features_names)
    X_new = input_data[features_names]

    # Scaling pour SVM
    X_new_scaled = scaler.transform(X_new)

    # --- Prédiction RF ---
    prediction_rf = rf_model.predict(X_new)
    proba_rf = rf_model.predict_proba(X_new)
    resultat_rf = le.inverse_transform(prediction_rf)[0]
    
    print(f"\n🌲 Avis du Random Forest :")
    print(f"👉 Résultat prévu : {traduire_resultat(resultat_rf)}")
    print(f"📊 Confiance :")
    for i, classe in enumerate(le.classes_):
        print(f"   - {traduire_resultat(classe)} : {proba_rf[0][i]*100:.1f}%")

    # --- Prédiction SVM ---
    prediction_svm = svm_model.predict(X_new_scaled)
    resultat_svm = le.inverse_transform(prediction_svm)[0]
    print(f"\n🛡️ Avis du SVM : {traduire_resultat(resultat_svm)}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Chargement & Nettoyage
    df = get_data()
    df_clean = clean_data(df)
    
    # 2. Entraînement et récupération des objets
    rf_model, svm_model, scaler, le = run_classification(df_clean)
    
    # 3. Régression (On utilise une équipe avec bcp de données)
    run_regression(df_clean, "Paris SG")
    
    # 4. Exemple de Prédiction (PSG vs OM)
    prochain_match = {
        'hometeam': 'Paris SG',
        'awayteam': 'Marseille',
        'cote_dom_clean': 1.55,       # PSG Favori
        'cote_nul_clean': 4.20,
        'cote_ext_clean': 6.00,
        'home_forme_pts_last5': 12,   # Bonne forme
        'away_forme_pts_last5': 8,    # Forme moyenne
        'home_moy_buts_marques_last5': 2.2,
        'away_moy_buts_encaisse_last5': 1.1
    }
    
    predire_un_match(rf_model, svm_model, scaler, le, prochain_match)