# Guide pour obtenir les bonnes clés Supabase

## Problème actuel
Votre clé `sb_publishable_-2Fofz2Gxy2hqabBgt5w0A_b7k2eihI` n'est plus acceptée par la bibliothèque Supabase Python 2.3.0.

## Solution : Obtenir les nouvelles clés

1. **Allez sur** https://supabase.com/dashboard
2. **Sélectionnez votre projet**
3. **Cliquez sur Settings (⚙️)** dans le menu de gauche
4. **Cliquez sur "API"** dans le sous-menu

## Clés disponibles

Vous verrez plusieurs clés :

### Option 1 : Clé "anon public" (recommandée pour lecture)
- **Format** : Commence par `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
- **Usage** : Pour les opérations de lecture (SELECT)
- **Sécurité** : Permissions limitées, sécurisée pour le frontend

### Option 2 : Clé "service_role" (pour écriture/administration)
- **Format** : Commence aussi par `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` mais plus longue
- **Usage** : Pour toutes les opérations (lecture, écriture, administration)
- **Sécurité** : ⚠️ **TRÈS SENSIBLE** - Ne jamais exposer au frontend !

## Quelle clé utiliser ?

Pour votre API FastAPI qui fait de la lecture de données :
- **Utilisez la clé "anon public"** si vous ne faites que lire des données
- **Utilisez la clé "service_role"** si vous avez besoin d'écrire des données

## Mise à jour du .env

Une fois que vous avez copié la bonne clé, mettez à jour votre `.env` :

```env
SUPABASE_URL=https://lqckcteuponqeisgovhr.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...  # Votre nouvelle clé
```

## Test

Après avoir mis à jour, testez avec :
```bash
python test_supabase_connection.py
```

Ou via l'API :
```bash
GET /health/supabase
```

