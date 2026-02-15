"""Service pour la gestion des données."""
import pandas as pd
import numpy as np
import httpx
from app.config import settings

class DataService:
    """Service pour récupérer et nettoyer les données."""
    
    def __init__(self):
        """Initialise le service."""
        self._supabase_url = settings.SUPABASE_URL.rstrip('/')
        self._supabase_key = settings.SUPABASE_KEY
        self._client = None
    
    @property
    def client(self) -> httpx.Client:
        """Récupère ou crée le client HTTP."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=f"{self._supabase_url}/rest/v1",
                headers={
                    "apikey": self._supabase_key,
                    "Authorization": f"Bearer {self._supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                },
                timeout=30.0
            )
        return self._client
    
    def get_data(self) -> pd.DataFrame:
        """Récupère les données de Supabase."""
        try:
            response = self.client.get("/ai_training_data", params={"select": "*"})
            response.raise_for_status()
            df = pd.DataFrame(response.json())
            return df
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Table 'ai_training_data' non trouvée. "
                    f"Vérifiez que la table existe dans votre projet Supabase."
                )
            raise ValueError(f"Erreur HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise ValueError(f"Erreur lors de la récupération des données: {str(e)}")
    
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

