"""Configuration de l'application."""
import os
from typing import Optional
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

class Settings:
    """Paramètres de configuration de l'API."""
    
    # Configuration Supabase
    SUPABASE_URL: str = os.getenv(
        "SUPABASE_URL", 
        "https://lqckcteuponqeisgovhr.supabase.co"
    )
    SUPABASE_KEY: str = os.getenv(
        "SUPABASE_KEY",
        "sb_publishable_-2Fofz2Gxy2hqabBgt5w0A_b7k2eihI"
    )
    
    # Configuration API
    API_TITLE: str = "API de Prédiction de Matchs de Football"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = """
    API pour la prédiction de résultats de matchs de football en utilisant 
    des modèles de machine learning (Random Forest et SVM).
    
    ## Fonctionnalités
    
    * **Entraînement** : Entraîner les modèles de classification
    * **Prédiction** : Prédire le résultat d'un match
    * **Régression** : Analyser l'évolution des cotes d'une équipe
    * **Analyse** : Analyser l'importance des features
    """
    
    # Configuration ML
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

settings = Settings()

