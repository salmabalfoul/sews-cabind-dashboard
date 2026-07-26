"""
Création du schéma de la base de données SQLite
Projet : Système intelligent de pilotage des faisceaux — SEWS Cabind

À lancer UNE SEULE FOIS avant le pipeline ETL.
"""

from sqlalchemy import (
    create_engine, text,
    Column, Integer, String, Float, Boolean, DateTime
)
from sqlalchemy.orm import declarative_base
from db_config import get_url
import os

Base = declarative_base()


class ProductionHarness(Base):
    """Une ligne = un faisceau produit."""
    __tablename__ = "production_harness"

    harness_id               = Column(Integer,  primary_key=True, autoincrement=False)
    reference                = Column(String,   nullable=False)
    date_production          = Column(DateTime, nullable=False)
    ligne_production         = Column(String,   nullable=False)
    operateur_id             = Column(String,   nullable=False)
    shift                    = Column(String,   nullable=False)
    nb_circuits              = Column(Integer,  nullable=False)
    nb_connecteurs           = Column(Integer,  nullable=False)
    longueur_totale_m        = Column(Float,    nullable=False)
    temps_cycle_min          = Column(Float,    nullable=False)
    statut_conformite        = Column(Boolean,  nullable=False)
    type_defaut              = Column(String,   nullable=False)
    phase_detection          = Column(String,   nullable=False)
    resultat_test_continuite         = Column(Boolean, nullable=False)
    resultat_test_court_circuit      = Column(Boolean, nullable=False)
    resistance_ohm           = Column(Float,    nullable=False)
    force_arrachement_N      = Column(Float,    nullable=False)
    hauteur_sertissage_mm    = Column(Float,    nullable=False)
    rebut                    = Column(Boolean,  nullable=False)
    retravail                = Column(Boolean,  nullable=False)
    temps_retravail_min      = Column(Float,    nullable=False)


def creer_base():
    print("=" * 55)
    print("  Création du schéma SQLite — SEWS Cabind")
    print("=" * 55)

    # Crée le dossier data/ si absent
    os.makedirs("data", exist_ok=True)

    engine = create_engine(get_url(), echo=False)

    # Crée toutes les tables
    Base.metadata.create_all(engine)
    print("  ✓ Table production_harness créée")

    # Index manuels pour accélérer les requêtes KPI
    index_sql = [
        "CREATE INDEX IF NOT EXISTS idx_reference       ON production_harness (reference);",
        "CREATE INDEX IF NOT EXISTS idx_date            ON production_harness (date_production);",
        "CREATE INDEX IF NOT EXISTS idx_shift           ON production_harness (shift);",
        "CREATE INDEX IF NOT EXISTS idx_type_defaut     ON production_harness (type_defaut);",
        "CREATE INDEX IF NOT EXISTS idx_statut          ON production_harness (statut_conformite);",
        "CREATE INDEX IF NOT EXISTS idx_ligne           ON production_harness (ligne_production);",
        "CREATE INDEX IF NOT EXISTS idx_operateur       ON production_harness (operateur_id);",
    ]

    with engine.connect() as conn:
        for sql in index_sql:
            conn.execute(text(sql))
        conn.commit()
    print("  ✓ Index créés")

    # Vérification
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='index';"
        )).fetchall()
    print(f"  ✓ {len(result)} index présents dans la base")

    print("\n  Fichier créé : data/sews_production.db")
    print("  Lance maintenant : python etl_pipeline.py")
    print("=" * 55)
    return engine


if __name__ == "__main__":
    creer_base()