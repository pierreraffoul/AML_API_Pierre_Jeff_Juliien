#!/usr/bin/env python3
"""Script pour démarrer l'API FastAPI."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Recharge automatique en cas de modification du code
    )

