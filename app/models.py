"""Modèles Pydantic pour la validation des données."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class MatchPredictionRequest(BaseModel):
    """Requête pour prédire un match."""
    hometeam: str = Field(..., description="Nom de l'équipe à domicile")
    awayteam: str = Field(..., description="Nom de l'équipe à l'extérieur")
    cote_dom_clean: float = Field(..., description="Cote de victoire domicile", gt=0)
    cote_nul_clean: float = Field(..., description="Cote de match nul", gt=0)
    cote_ext_clean: float = Field(..., description="Cote de victoire extérieur", gt=0)
    home_forme_pts_last5: float = Field(..., description="Points de forme domicile (5 derniers matchs)", ge=0)
    away_forme_pts_last5: float = Field(..., description="Points de forme extérieur (5 derniers matchs)", ge=0)
    home_moy_buts_marques_last5: float = Field(..., description="Moyenne de buts marqués domicile (5 derniers)", ge=0)
    away_moy_buts_encaisse_last5: float = Field(..., description="Moyenne de buts encaissés extérieur (5 derniers)", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "hometeam": "Paris SG",
                "awayteam": "Marseille",
                "cote_dom_clean": 1.55,
                "cote_nul_clean": 4.20,
                "cote_ext_clean": 6.00,
                "home_forme_pts_last5": 12.0,
                "away_forme_pts_last5": 8.0,
                "home_moy_buts_marques_last5": 2.2,
                "away_moy_buts_encaisse_last5": 1.1
            }
        }

class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    hometeam: str
    awayteam: str
    random_forest: Dict[str, Any] = Field(..., description="Prédiction du modèle Random Forest")
    svm: Dict[str, Any] = Field(..., description="Prédiction du modèle SVM")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hometeam": "Paris SG",
                "awayteam": "Marseille",
                "random_forest": {
                    "prediction": "H",
                    "probabilities": {
                        "H": 0.65,
                        "D": 0.20,
                        "A": 0.15
                    },
                    "prediction_text": "Victoire Domicile 🏠"
                },
                "svm": {
                    "prediction": "H",
                    "prediction_text": "Victoire Domicile 🏠"
                }
            }
        }

class RegressionRequest(BaseModel):
    """Requête pour l'analyse de régression."""
    team_name: str = Field(..., description="Nom de l'équipe à analyser")
    
    class Config:
        json_schema_extra = {
            "example": {
                "team_name": "Paris SG"
            }
        }

class RegressionResponse(BaseModel):
    """Réponse de régression."""
    team_name: str
    coefficient: float = Field(..., description="Coefficient de la pente de régression")
    trend: str = Field(..., description="Tendance (en hausse/en baisse)")
    message: str = Field(..., description="Message explicatif")
    data_points: int = Field(..., description="Nombre de points de données utilisés")
    chart_filename: Optional[str] = Field(None, description="Nom du fichier du graphique généré")

class ClassificationMetrics(BaseModel):
    """Métriques de classification."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: List[List[int]]
    labels: List[str]

class ModelTrainingResponse(BaseModel):
    """Réponse après entraînement des modèles."""
    status: str
    random_forest: ClassificationMetrics
    svm: ClassificationMetrics
    best_params: Optional[Dict[str, Any]] = None
    auc_score: Optional[float] = None

class FeatureImportanceResponse(BaseModel):
    """Réponse pour l'analyse d'importance des features."""
    features: List[Dict[str, Any]] = Field(..., description="Liste des features avec leur importance")
    chart_filename: Optional[str] = Field(None, description="Nom du fichier du graphique généré")

class HealthResponse(BaseModel):
    """Réponse de santé de l'API."""
    status: str
    version: str

