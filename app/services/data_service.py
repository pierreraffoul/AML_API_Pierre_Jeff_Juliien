"""Service pour la gestion des données."""
import pandas as pd
import numpy as np
from supabase import create_client, Client
from app.config import settings

class DataService:
    """Service pour récupérer et nettoyer les données."""
    
    def __init__(self):
        """Initialise la connexion Supabase."""
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    def get_data(self) -> pd.DataFrame:
        """Récupère les données de Supabase."""
        response = self.supabase.table("ai_training_data").select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données (erreurs de types, dates dans colonnes numériques)."""
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
        df = df.dropna(subset=numeric_cols + ['ftr'])
        
        return df
    
    def get_and_clean_data(self) -> pd.DataFrame:
        """Récupère et nettoie les données en une seule opération."""
        df = self.get_data()
        return self.clean_data(df)

