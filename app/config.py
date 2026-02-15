"""Configuration de l'application."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
# Chercher le fichier .env à la racine du projet
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    """Paramètres de configuration de l'API."""
    
    def __init__(self):
        """Initialise et valide les paramètres."""
        # Configuration Supabase
        self.SUPABASE_URL: str = os.getenv(
            "SUPABASE_URL", 
            "https://lqckcteuponqeisgovhr.supabase.co"
        )
        self.SUPABASE_KEY: str = os.getenv(
            "SUPABASE_KEY",
            "sb_publishable_-2Fofz2Gxy2hqabBgt5w0A_b7k2eihI"
        )
        
        # Valider que les clés sont bien définies
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError(
                "Les clés Supabase (SUPABASE_URL et SUPABASE_KEY) doivent être définies "
                "dans le fichier .env ou comme variables d'environnement."
            )
        if self.SUPABASE_KEY == "votre-clé-supabase" or self.SUPABASE_URL == "https://votre-url.supabase.co":
            raise ValueError(
                "Veuillez configurer vos vraies clés Supabase dans le fichier .env"
            )
        
        # Configuration API
        self.API_TITLE: str = "API de Prédiction de Matchs de Football"
        self.API_VERSION: str = "1.0.0"
        self.API_DESCRIPTION: str = """
        API pour la prédiction de résultats de matchs de football en utilisant 
        des modèles de machine learning (Random Forest et SVM).
        
        ## Fonctionnalités
        
        * **Entraînement** : Entraîner les modèles de classification
        * **Prédiction** : Prédire le résultat d'un match
        * **Régression** : Analyser l'évolution des cotes d'une équipe
        * **Analyse** : Analyser l'importance des features
        """
        
        # Configuration ML
        self.TEST_SIZE: float = 0.2
        self.RANDOM_STATE: int = 42

# Créer l'instance settings avec validation
settings = Settings()

