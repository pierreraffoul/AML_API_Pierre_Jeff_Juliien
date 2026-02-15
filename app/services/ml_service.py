"""Service pour les modèles de machine learning."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour la production
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any, Optional
import os
import pickle
from pathlib import Path

from app.config import settings

class MLService:
    """Service pour l'entraînement et l'utilisation des modèles ML."""
    
    # Dossier pour sauvegarder les modèles
    MODELS_DIR = Path("models")
    
    def __init__(self):
        """Initialise le service ML et charge les modèles s'ils existent."""
        self.rf_model: Optional[RandomForestClassifier] = None
        self.svm_model: Optional[SVC] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.features = [
            'cote_dom_clean', 'cote_nul_clean', 'cote_ext_clean',
            'home_forme_pts_last5', 'away_forme_pts_last5',
            'home_moy_buts_marques_last5', 'away_moy_buts_encaisse_last5'
        ]
        
        # Créer le dossier models s'il n'existe pas
        self.MODELS_DIR.mkdir(exist_ok=True)
        
        # Charger automatiquement les modèles s'ils existent
        self.load_models()
    
    def is_trained(self) -> bool:
        """Vérifie si les modèles sont entraînés."""
        return (
            self.rf_model is not None and 
            self.svm_model is not None and 
            self.scaler is not None and 
            self.label_encoder is not None
        )
    
    def save_models(self) -> bool:
        """Sauvegarde les modèles entraînés."""
        if not self.is_trained():
            return False
        
        try:
            # Sauvegarder chaque composant
            with open(self.MODELS_DIR / "rf_model.pkl", "wb") as f:
                pickle.dump(self.rf_model, f)
            with open(self.MODELS_DIR / "svm_model.pkl", "wb") as f:
                pickle.dump(self.svm_model, f)
            with open(self.MODELS_DIR / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            with open(self.MODELS_DIR / "label_encoder.pkl", "wb") as f:
                pickle.dump(self.label_encoder, f)
            
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des modèles : {e}")
            return False
    
    def load_models(self) -> bool:
        """Charge les modèles sauvegardés."""
        try:
            rf_path = self.MODELS_DIR / "rf_model.pkl"
            svm_path = self.MODELS_DIR / "svm_model.pkl"
            scaler_path = self.MODELS_DIR / "scaler.pkl"
            encoder_path = self.MODELS_DIR / "label_encoder.pkl"
            
            # Vérifier que tous les fichiers existent
            if not all([rf_path.exists(), svm_path.exists(), scaler_path.exists(), encoder_path.exists()]):
                return False
            
            # Charger chaque composant
            with open(rf_path, "rb") as f:
                self.rf_model = pickle.load(f)
            with open(svm_path, "rb") as f:
                self.svm_model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des modèles : {e}")
            return False
    
    def train_classification_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entraîne les modèles de classification."""
        X = df[self.features]
        y = df['ftr']
        
        # Encodage de la cible
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        labels = self.label_encoder.classes_
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=settings.TEST_SIZE, 
            random_state=settings.RANDOM_STATE
        )
        
        # Scaling pour SVM
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # GridSearch pour Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=settings.RANDOM_STATE), 
            param_grid, 
            cv=3, 
            scoring='roc_auc_ovr'
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        # Entraînement Random Forest avec paramètres optimisés
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=settings.RANDOM_STATE
        )
        self.rf_model.fit(X_train, y_train)
        y_pred_rf = self.rf_model.predict(X_test)
        proba_rf = self.rf_model.predict_proba(X_test)
        
        # Entraînement SVM
        self.svm_model = SVC(
            kernel='rbf', 
            C=1.0, 
            probability=True, 
            class_weight='balanced', 
            random_state=settings.RANDOM_STATE
        )
        self.svm_model.fit(X_train_scaled, y_train)
        y_pred_svm = self.svm_model.predict(X_test_scaled)
        
        # Calcul des métriques
        rf_metrics = self._calculate_metrics(y_test, y_pred_rf, labels)
        svm_metrics = self._calculate_metrics(y_test, y_pred_svm, labels)
        
        # Calcul AUC
        auc_score = None
        try:
            auc_score = roc_auc_score(y_test, proba_rf, multi_class='ovr')
        except Exception:
            pass
        
        # Génération des matrices de confusion
        self._plot_confusion_matrix(y_test, y_pred_rf, labels, "Matrice RF Optimisée", "confusion_matrix_rf.png")
        self._plot_confusion_matrix(y_test, y_pred_svm, labels, "Matrice SVM Optimisée", "confusion_matrix_svm.png")
        
        # Sauvegarder automatiquement les modèles après l'entraînement
        self.save_models()
        
        return {
            'random_forest': rf_metrics,
            'svm': svm_metrics,
            'best_params': best_params,
            'auc_score': auc_score
        }
    
    def _calculate_metrics(self, y_true, y_pred, labels) -> Dict[str, Any]:
        """Calcule les métriques de classification."""
        # Calcul des métriques par classe
        precision_per_class = precision_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': {label: float(precision_per_class[i]) for i, label in enumerate(labels)},
            'recall': {label: float(recall_per_class[i]) for i, label in enumerate(labels)},
            'f1_score': {label: float(f1_per_class[i]) for i, label in enumerate(labels)},
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'labels': labels.tolist()
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, labels, title, filename):
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
    
    def predict_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit le résultat d'un match."""
        if not self.is_trained():
            raise ValueError(
                "Les modèles doivent être entraînés avant de faire des prédictions. "
                "Utilisez l'endpoint POST /train pour entraîner les modèles."
            )
        
        # Création DataFrame
        input_data = pd.DataFrame([match_data], columns=['hometeam', 'awayteam'] + self.features)
        X_new = input_data[self.features]
        
        # Scaling pour SVM
        X_new_scaled = self.scaler.transform(X_new)
        
        # Prédiction RF
        prediction_rf = self.rf_model.predict(X_new)
        proba_rf = self.rf_model.predict_proba(X_new)
        resultat_rf = self.label_encoder.inverse_transform(prediction_rf)[0]
        
        # Prédiction SVM
        prediction_svm = self.svm_model.predict(X_new_scaled)
        resultat_svm = self.label_encoder.inverse_transform(prediction_svm)[0]
        
        # Formatage des probabilités
        probabilities = {}
        for i, classe in enumerate(self.label_encoder.classes_):
            probabilities[classe] = float(proba_rf[0][i])
        
        return {
            'random_forest': {
                'prediction': resultat_rf,
                'probabilities': probabilities,
                'prediction_text': self._traduire_resultat(resultat_rf)
            },
            'svm': {
                'prediction': resultat_svm,
                'prediction_text': self._traduire_resultat(resultat_svm)
            }
        }
    
    def _traduire_resultat(self, code: str) -> str:
        """Traduit H/D/A en texte lisible."""
        if code == 'H': return "Victoire Domicile 🏠"
        if code == 'A': return "Victoire Extérieur ✈️"
        return "Match Nul 🤝"
    
    def analyze_regression(self, df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
        """Analyse l'évolution des cotes d'une équipe."""
        # Filtrer l'équipe
        team_df = df[(df['hometeam'] == team_name) | (df['awayteam'] == team_name)].copy()
        
        if len(team_df) < 10:
            raise ValueError(f"Pas assez de données pour {team_name}. Minimum 10 matchs requis.")
        
        # Préparer la cote à analyser
        team_df['ma_cote'] = np.where(
            team_df['hometeam'] == team_name, 
            team_df['cote_dom_clean'], 
            team_df['cote_ext_clean']
        )
        
        team_df = team_df.sort_values('date')
        team_df['time_index'] = np.arange(len(team_df))
        
        X = team_df[['time_index']]
        y = team_df['ma_cote']
        
        # Entraînement
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        # Analyse
        coef = float(reg.coef_[0])
        tendance = "en hausse ↗️" if coef > 0 else "en baisse ↘️"
        msg_confiance = "(L'équipe est moins favorite)" if coef > 0 else "(L'équipe est plus favorite)"
        
        # Sauvegarde du graphique
        filename = f"regression_{team_name.replace(' ', '_')}.png"
        plt.figure(figsize=(10, 6))
        plt.scatter(team_df['date'], y, color='blue', alpha=0.4, label='Cotes réelles')
        plt.plot(team_df['date'], y_pred, color='red', linewidth=2, label='Tendance')
        plt.title(f"Évolution des cotes de victoire : {team_name}")
        plt.xlabel("Années")
        plt.ylabel("Cote")
        plt.legend()
        plt.savefig(filename)
        plt.close()
        
        return {
            'team_name': team_name,
            'coefficient': coef,
            'trend': tendance,
            'message': f"Les cotes de {team_name} sont globalement {tendance} {msg_confiance}.",
            'data_points': len(team_df),
            'chart_filename': filename
        }
    
    def analyze_feature_importance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse l'importance des features."""
        feature_names_clean = [
            'Cote Domicile', 
            'Cote Nul', 
            'Cote Extérieur',
            'Forme Dom (5 derniers)', 
            'Forme Ext (5 derniers)',
            'Attaque Domicile', 
            'Défense Extérieur'
        ]
        
        X = df[self.features]
        y = df['ftr']
        
        # Encodage de la cible
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Entraînement Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=5, 
            random_state=settings.RANDOM_STATE
        )
        rf_model.fit(X, y_encoded)
        
        # Extraction des importances
        importances = rf_model.feature_importances_
        
        # Création d'un DataFrame pour trier les résultats
        feature_imp_df = pd.DataFrame({
            'Variable': feature_names_clean,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Visualisation
        filename = "feature_importance_analysis.png"
        plt.figure(figsize=(10, 6))
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
        plt.savefig(filename)
        plt.close()
        
        # Formatage des résultats
        features_list = [
            {
                'name': row['Variable'],
                'importance': float(row['Importance'])
            }
            for _, row in feature_imp_df.iterrows()
        ]
        
        return {
            'features': features_list,
            'chart_filename': filename
        }

