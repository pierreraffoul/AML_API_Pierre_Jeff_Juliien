# API de Prédiction de Matchs de Football

API REST construite avec FastAPI pour prédire les résultats de matchs de football en utilisant des modèles de machine learning (Random Forest et SVM).

## 🚀 Fonctionnalités

- **Entraînement des modèles** : Entraînement des modèles de classification (Random Forest et SVM)
- **Prédiction de matchs** : Prédiction du résultat d'un match (Victoire Domicile, Match Nul, Victoire Extérieur)
- **Analyse de régression** : Analyse de l'évolution des cotes d'une équipe dans le temps
- **Analyse d'importance** : Analyse de l'importance des différentes features dans la prédiction
- **Affichage des métriques** : Script indépendant pour afficher les métriques détaillées des modèles
- **Documentation interactive** : Documentation automatique avec Swagger UI et ReDoc

## 📋 Prérequis

- Python 3.8+
- Accès à une base de données Supabase (ou modification de la configuration)

## 🔧 Installation

1. **Cloner le dépôt** (si applicable) ou naviguer vers le répertoire du projet

2. **Créer un environnement virtuel** (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

3. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement** (optionnel) :
Créer un fichier `.env` à la racine du projet :
```env
SUPABASE_URL=https://votre-url.supabase.co
SUPABASE_KEY=votre-clé-supabase
```

Sinon, les valeurs par défaut dans `app/config.py` seront utilisées.

## 🚀 Utilisation

### Script d'affichage des métriques (Indépendant)

Pour afficher les métriques détaillées des modèles SVM et Random Forest sans utiliser l'API :

```bash
# Activer l'environnement virtuel
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows

# Exécuter le script
python display_metrics.py
```

Ce script va :
- Récupérer les données depuis Supabase
- Entraîner les modèles Random Forest et SVM
- Afficher les métriques détaillées (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Générer les matrices de confusion (`confusion_matrix_rf.png` et `confusion_matrix_svm.png`)
- Comparer les performances des deux modèles

**Métriques affichées** :
- Accuracy (précision globale)
- Precision, Recall, F1-Score par classe (Domicile, Nul, Extérieur)
- Métriques moyennes (macro)
- AUC-ROC (Area Under the Curve)
- Matrices de confusion
- Rapport de classification détaillé

### Démarrer l'API

```bash
uvicorn app.main:app --reload
```

Ou utiliser le script de démarrage :
```bash
python run_api.py
```

L'API sera accessible à l'adresse : `http://localhost:8000`

### Documentation interactive

Une fois l'API démarrée, accédez à :

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## 📚 Endpoints de l'API

### 1. Santé de l'API

#### `GET /`
Vérifie que l'API fonctionne.

**Réponse** :
```json
{
  "status": "OK",
  "version": "1.0.0"
}
```

#### `GET /health`
Vérifie l'état de santé de l'API.

#### `GET /health/supabase`
Vérifie la connexion à Supabase et teste si les clés sont valides.

**Réponse** :
```json
{
  "status": "OK",
  "message": "Connexion Supabase réussie",
  "data_count": 5321
}
```

#### `GET /models/status`
Vérifie si les modèles sont entraînés et disponibles.

**Réponse** :
```json
{
  "trained": true,
  "message": "Les modèles sont entraînés et prêts à être utilisés."
}
```

### 2. Entraînement

#### `POST /train`
Entraîne les modèles de classification (Random Forest et SVM).

**Réponse** :
```json
{
  "status": "success",
  "random_forest": {
    "accuracy": 0.85,
    "precision": {"H": 0.88, "D": 0.75, "A": 0.82},
    "recall": {"H": 0.90, "D": 0.70, "A": 0.80},
    "f1_score": {"H": 0.89, "D": 0.72, "A": 0.81},
    "confusion_matrix": [[...], [...], [...]],
    "labels": ["H", "D", "A"]
  },
  "svm": { ... },
  "best_params": { ... },
  "auc_score": 0.92
}
```

**Note** : 
- Cette opération peut prendre plusieurs minutes
- Les modèles sont automatiquement sauvegardés dans le dossier `models/` après l'entraînement
- Les modèles sauvegardés sont automatiquement rechargés au redémarrage de l'API

### 3. Prédiction

#### `POST /predict`
Prédit le résultat d'un match.

**Corps de la requête** :
```json
{
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
```

**Réponse** :
```json
{
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
```

**Important** : Les modèles doivent être entraînés via `/train` avant d'utiliser cet endpoint.

### 4. Analyse

#### `POST /regression`
Analyse l'évolution des cotes d'une équipe dans le temps.

**Corps de la requête** :
```json
{
  "team_name": "Paris SG"
}
```

**Réponse** :
```json
{
  "team_name": "Paris SG",
  "coefficient": -0.05,
  "trend": "en baisse ↘️",
  "message": "Les cotes de Paris SG sont globalement en baisse ↘️ (L'équipe est plus favorite).",
  "data_points": 45,
  "chart_filename": "regression_Paris_SG.png"
}
```

#### `GET /feature-importance`
Analyse l'importance des différentes features dans la prédiction.

**Réponse** :
```json
{
  "features": [
    {
      "name": "Cote Domicile",
      "importance": 0.35
    },
    {
      "name": "Cote Extérieur",
      "importance": 0.28
    },
    ...
  ],
  "chart_filename": "feature_importance_analysis.png"
}
```

### 5. Ressources

#### `GET /charts/{filename}`
Récupère un graphique généré par l'API.

**Graphiques disponibles** :
- `confusion_matrix_rf.png` : Matrice de confusion Random Forest
- `confusion_matrix_svm.png` : Matrice de confusion SVM
- `regression_{team_name}.png` : Graphique de régression pour une équipe
- `feature_importance_analysis.png` : Graphique d'importance des features

## 🏗️ Structure du projet

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # Point d'entrée FastAPI
│   ├── config.py            # Configuration de l'application
│   ├── models.py            # Modèles Pydantic pour la validation
│   └── services/
│       ├── __init__.py
│       ├── data_service.py  # Service de gestion des données
│       └── ml_service.py    # Service de machine learning
├── display_metrics.py       # Script indépendant pour afficher les métriques
├── run_api.py              # Script de démarrage de l'API
├── main.py                 # Script original (ancien code)
├── t.py                    # Script d'analyse d'importance
├── requirements.txt        # Dépendances Python
├── .env                    # Variables d'environnement (non versionné)
├── .env.example            # Exemple de fichier .env
└── README.md              # Ce fichier
```

## 📊 Scripts disponibles

### `display_metrics.py`
Script Python indépendant qui :
- Récupère les données depuis Supabase
- Entraîne les modèles Random Forest et SVM
- Affiche toutes les métriques détaillées
- Génère des graphiques de matrices de confusion
- Compare les performances des modèles

**Utilisation** :
```bash
python display_metrics.py
```

### `run_api.py`
Script de démarrage simplifié pour l'API FastAPI.

**Utilisation** :
```bash
python run_api.py
```

## 🔒 Sécurité

⚠️ **Important** : En production, modifiez les paramètres suivants :

1. **CORS** : Dans `app/main.py`, remplacez `allow_origins=["*"]` par les origines autorisées
2. **Variables d'environnement** : Utilisez des variables d'environnement pour les clés Supabase
3. **HTTPS** : Utilisez HTTPS en production

## 🧪 Exemples d'utilisation

### Avec cURL

**Entraîner les modèles** :
```bash
curl -X POST "http://localhost:8000/train"
```

**Prédire un match** :
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hometeam": "Paris SG",
    "awayteam": "Marseille",
    "cote_dom_clean": 1.55,
    "cote_nul_clean": 4.20,
    "cote_ext_clean": 6.00,
    "home_forme_pts_last5": 12.0,
    "away_forme_pts_last5": 8.0,
    "home_moy_buts_marques_last5": 2.2,
    "away_moy_buts_encaisse_last5": 1.1
  }'
```

### Avec Python

```python
import requests

# Entraîner les modèles
response = requests.post("http://localhost:8000/train")
print(response.json())

# Prédire un match
match_data = {
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
response = requests.post("http://localhost:8000/predict", json=match_data)
print(response.json())
```

## 📝 Notes

- Les modèles doivent être entraînés avant de faire des prédictions
- Les graphiques sont sauvegardés dans le répertoire courant
- L'API utilise des modèles de machine learning qui nécessitent des données propres et complètes
- Le script `display_metrics.py` est indépendant de l'API et peut être utilisé pour analyser les modèles sans démarrer le serveur
- Les modèles entraînés via l'API sont sauvegardés dans le dossier `models/` pour être réutilisés au redémarrage

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📄 Licence

Ce projet est fourni tel quel, sans garantie.

# AML_API_Pierre_Jeff_Juliien
