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
# Note: DataService initialise la connexion Supabase de manière paresseuse
# pour permettre à l'API de démarrer même si la connexion échoue
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


@app.get("/health/supabase", tags=["Santé"])
async def health_check_supabase():
    """
    Vérifie la connexion à Supabase.
    
    Teste si les clés Supabase sont valides et si la connexion fonctionne.
    """
    try:
        df = data_service.get_data()
        return {
            "status": "OK",
            "message": "Connexion Supabase réussie",
            "data_count": len(df)
        }
    except ValueError as e:
        error_msg = str(e)
        if "Supabase" in error_msg or "API key" in error_msg:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Erreur de connexion à Supabase. "
                    "Vérifiez que vos clés SUPABASE_URL et SUPABASE_KEY sont correctes dans le fichier .env. "
                    "Assurez-vous d'utiliser la clé 'anon public' depuis votre tableau de bord Supabase."
                )
            )
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la vérification Supabase : {str(e)}"
        )


@app.get("/models/status", tags=["Entraînement"])
async def get_models_status():
    """
    Vérifie si les modèles sont entraînés et disponibles.
    
    Retourne le statut des modèles (entraînés ou non).
    """
    is_trained = ml_service.is_trained()
    return {
        "trained": is_trained,
        "message": "Les modèles sont entraînés et prêts à être utilisés." if is_trained 
                  else "Les modèles ne sont pas encore entraînés. Utilisez POST /train pour les entraîner."
    }


@app.post("/train", response_model=ModelTrainingResponse, tags=["Entraînement"])
async def train_models():
    """
    Entraîne les modèles de classification (Random Forest et SVM).
    
    Cette opération peut prendre plusieurs minutes selon la taille des données.
    Les modèles sont automatiquement sauvegardés après l'entraînement.
    """
    try:
        # Récupération et nettoyage des données
        df = data_service.get_and_clean_data()
        
        if df.empty:
            raise HTTPException(
                status_code=400, 
                detail="Aucune donnée disponible pour l'entraînement. Vérifiez votre connexion Supabase."
            )
        
        # Entraînement des modèles
        results = ml_service.train_classification_models(df)
        
        return {
            "status": "success",
            "random_forest": results['random_forest'],
            "svm": results['svm'],
            "best_params": results.get('best_params'),
            "auc_score": results.get('auc_score'),
            "message": "Modèles entraînés et sauvegardés avec succès."
        }
    except ValueError as e:
        # Erreur de connexion Supabase ou autres erreurs de validation
        error_msg = str(e)
        # Les messages d'erreur détaillés sont déjà dans ValueError
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'entraînement : {str(e)}"
        )


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

