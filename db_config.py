"""
Configuration de connexion — SQLite
Projet : Système intelligent de pilotage des faisceaux — SEWS Cabind

SQLite est intégré dans Python : zéro installation nécessaire.
Le fichier de base de données sera créé automatiquement dans data/
"""

import os

# Chemin vers le fichier SQLite (créé automatiquement s'il n'existe pas)
DB_PATH      = os.path.join("data", "sews_production.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"


def get_url():
    """Retourne l'URL de connexion SQLAlchemy."""
    return DATABASE_URL


if __name__ == "__main__":
    print("Configuration SQLite :")
    print(f"  Fichier base de données : {DB_PATH}")
    print(f"  URL SQLAlchemy          : {DATABASE_URL}")
    print(f"  Fichier existe          : {os.path.exists(DB_PATH)}")