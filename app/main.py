"""Point d'entrée de l'API FastAPI."""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from app.config import settings
from app.models import (
    MatchPredictionRequest,
    PredictionResponse,
    RegressionRequest,
    RegressionResponse,
    ModelTrainingResponse,
    FeatureImportanceResponse,
    HealthResponse
)
from app.services.data_service import DataService
from app.services.ml_service import MLService

# Initialisation de l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour permettre les requêtes depuis un client web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instances des services (singletons)
data_service = DataService()
ml_service = MLService()


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/", response_model=HealthResponse, tags=["Santé"])
async def root():
    """Endpoint racine pour vérifier que l'API fonctionne."""
    return {
        "status": "OK",
        "version": settings.API_VERSION
    }


@app.get("/health", response_model=HealthResponse, tags=["Santé"])
async def health_check():
    """Vérifie l'état de santé de l'API."""
    return {
        "status": "OK",
        "version": settings.API_VERSION
    }


@app.post("/train", response_model=ModelTrainingResponse, tags=["Entraînement"])
async def train_models():
    """
    Entraîne les modèles de classification (Random Forest et SVM).
    
    Cette opération peut prendre plusieurs minutes selon la taille des données.
    """
    try:
        # Récupération et nettoyage des données
        df = data_service.get_and_clean_data()
        
        # Entraînement des modèles
        results = ml_service.train_classification_models(df)
        
        return {
            "status": "success",
            "random_forest": results['random_forest'],
            "svm": results['svm'],
            "best_params": results.get('best_params'),
            "auc_score": results.get('auc_score')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement : {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_match(request: MatchPredictionRequest):
    """
    Prédit le résultat d'un match en utilisant les modèles entraînés.
    
    **Important** : Les modèles doivent être entraînés via `/train` avant d'utiliser cet endpoint.
    """
    try:
        match_data = request.dict()
        prediction = ml_service.predict_match(match_data)
        
        return {
            "hometeam": request.hometeam,
            "awayteam": request.awayteam,
            "random_forest": prediction['random_forest'],
            "svm": prediction['svm']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")


@app.post("/regression", response_model=RegressionResponse, tags=["Analyse"])
async def analyze_regression(request: RegressionRequest):
    """
    Analyse l'évolution des cotes d'une équipe dans le temps.
    
    Génère un graphique montrant la tendance des cotes de l'équipe.
    """
    try:
        df = data_service.get_and_clean_data()
        result = ml_service.analyze_regression(df, request.team_name)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")


@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Analyse"])
async def analyze_feature_importance():
    """
    Analyse l'importance des différentes features dans la prédiction.
    
    Génère un graphique montrant quelles variables sont les plus importantes
    pour les prédictions du modèle.
    """
    try:
        df = data_service.get_and_clean_data()
        result = ml_service.analyze_feature_importance(df)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")


@app.get("/charts/{filename}", tags=["Ressources"])
async def get_chart(filename: str):
    """
    Récupère un graphique généré par l'API.
    
    Les graphiques disponibles incluent :
    - `confusion_matrix_rf.png` : Matrice de confusion Random Forest
    - `confusion_matrix_svm.png` : Matrice de confusion SVM
    - `regression_{team_name}.png` : Graphique de régression pour une équipe
    - `feature_importance_analysis.png` : Graphique d'importance des features
    """
    try:
        return FileResponse(
            filename,
            media_type="image/png",
            filename=filename
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Graphique {filename} non trouvé")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

