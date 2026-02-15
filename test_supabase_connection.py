#!/usr/bin/env python3
"""Script de test pour vérifier la connexion Supabase."""
import os
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client

# Charger le .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

print("=" * 60)
print("TEST DE CONNEXION SUPABASE")
print("=" * 60)
print(f"\n📍 URL: {URL}")
print(f"🔑 Clé (premiers 30 caractères): {KEY[:30] if KEY else 'None'}...")
print(f"📏 Longueur de la clé: {len(KEY) if KEY else 0} caractères")

# Vérifier le format de la clé
print(f"\n🔍 Analyse de la clé:")
if KEY:
    if KEY.startswith("eyJ"):
        print("  ✅ Format JWT détecté (correct)")
    elif KEY.startswith("sb_"):
        print("  ❌ Format 'sb_publishable_' détecté (INCORRECT)")
        print("  ⚠️  Cette clé n'est pas valide pour l'API Supabase Python")
        print("  💡 Vous devez utiliser la clé 'anon public' depuis votre dashboard")
    else:
        print("  ⚠️  Format inconnu")
else:
    print("  ❌ Aucune clé trouvée")

print("\n" + "=" * 60)
print("TENTATIVE DE CONNEXION...")
print("=" * 60)

try:
    client = create_client(URL, KEY)
    print("✅ Connexion réussie!")
    
    # Tester une requête simple
    try:
        response = client.table("ai_training_data").select("id").limit(1).execute()
        print(f"✅ Test de requête réussi! ({len(response.data)} résultat(s))")
    except Exception as e:
        print(f"⚠️  Connexion OK mais erreur sur la requête: {e}")
        
except Exception as e:
    print(f"\n❌ ERREUR DE CONNEXION:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    
    if "Invalid API key" in str(e):
        print("\n" + "=" * 60)
        print("🔧 SOLUTION:")
        print("=" * 60)
        print("""
1. Allez sur https://supabase.com/dashboard
2. Sélectionnez votre projet
3. Cliquez sur "Settings" (⚙️) dans le menu de gauche
4. Cliquez sur "API" dans le sous-menu
5. Dans la section "Project API keys", copiez la clé "anon public"
   (Elle commence par "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
6. Remplacez SUPABASE_KEY dans votre fichier .env par cette clé
7. Redémarrez l'API

⚠️  NE PAS utiliser:
   - La clé "service_role" (trop permissive)
   - La clé "sb_publishable_..." (format incorrect)
   - Toute autre clé que "anon public"
        """)

print("\n" + "=" * 60)

