"""
Pipeline ETL — Extract / Transform / Load
Projet : Système intelligent de pilotage des faisceaux — SEWS Cabind
Base de données : SQLite
"""

import pandas as pd
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from db_config import get_url

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CSV_PATH   = os.path.join("data", "production_harness.csv")
CHUNK_SIZE = 500

VALEURS_VALIDES = {
    "shift": ["matin", "apres_midi", "nuit"],
    "type_defaut": [
        "circuit_ouvert", "court_circuit", "mauvais_sertissage",
        "isolation_endommagee", "mauvais_connecteur", "sortie_terminal",
        "erreur_cablage", "clip_manquant", "conforme"
    ],
    "phase_detection": [
        "sertissage", "test_electrique", "controle_visuel",
        "assemblage", "non_applicable"
    ],
}

PLAGES_PHYSIQUES = {
    "hauteur_sertissage_mm": (0.5,   4.0),
    "force_arrachement_N":   (5.0, 200.0),
    "resistance_ohm":        (0.0,   0.5),
    "temps_cycle_min":       (1.0, 300.0),
    "longueur_totale_m":     (0.1,  20.0),
}


# ═══════════════════════════════════════════
# ÉTAPE 1 — EXTRACT
# ═══════════════════════════════════════════
def extract(chemin_csv: str) -> pd.DataFrame:
    print("\n── EXTRACT ─────────────────────────────────")
    print(f"  Lecture de : {chemin_csv}")

    if not os.path.exists(chemin_csv):
        print(f"  ✗ Fichier introuvable : {chemin_csv}")
        print("    Lance d'abord : python generate_data.py")
        sys.exit(1)

    df = pd.read_csv(chemin_csv)
    print(f"  ✓ {len(df):,} lignes lues · {df.shape[1]} colonnes")

    colonnes_attendues = [
        "harness_id", "reference", "date_production", "ligne_production",
        "operateur_id", "shift", "nb_circuits", "nb_connecteurs",
        "longueur_totale_m", "temps_cycle_min", "statut_conformite",
        "type_defaut", "phase_detection", "resultat_test_continuite",
        "resultat_test_court_circuit", "resistance_ohm",
        "force_arrachement_N", "hauteur_sertissage_mm",
        "rebut", "retravail", "temps_retravail_min",
    ]
    manquantes = [c for c in colonnes_attendues if c not in df.columns]
    if manquantes:
        print(f"  ✗ Colonnes manquantes : {manquantes}")
        sys.exit(1)

    print("  ✓ Toutes les colonnes sont présentes")
    return df


# ═══════════════════════════════════════════
# ÉTAPE 2 — TRANSFORM
# ═══════════════════════════════════════════
def transform(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── TRANSFORM ───────────────────────────────")
    nb_initial = len(df)

    # Typage
    df["date_production"]              = pd.to_datetime(df["date_production"])
    df["harness_id"]                   = df["harness_id"].astype(int)
    df["nb_circuits"]                  = df["nb_circuits"].astype(int)
    df["nb_connecteurs"]               = df["nb_connecteurs"].astype(int)
    df["statut_conformite"]            = df["statut_conformite"].astype(bool)
    df["resultat_test_continuite"]     = df["resultat_test_continuite"].astype(bool)
    df["resultat_test_court_circuit"]  = df["resultat_test_court_circuit"].astype(bool)
    df["rebut"]                        = df["rebut"].astype(bool)
    df["retravail"]                    = df["retravail"].astype(bool)

    for col in ["longueur_totale_m", "temps_cycle_min", "resistance_ohm",
                "force_arrachement_N", "hauteur_sertissage_mm", "temps_retravail_min"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["reference", "ligne_production", "operateur_id",
                "shift", "type_defaut", "phase_detection"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    print("  ✓ Typage effectué")

    # Doublons
    nb_avant = len(df)
    df = df.drop_duplicates(subset=["harness_id"])
    nb_doublons = nb_avant - len(df)
    print(f"  ✓ Doublons supprimés : {nb_doublons}")

    # Valeurs manquantes
    df = df.dropna(subset=["harness_id", "reference", "date_production",
                            "statut_conformite", "type_defaut"])
    for col in ["longueur_totale_m", "temps_cycle_min", "resistance_ohm",
                "force_arrachement_N", "hauteur_sertissage_mm", "temps_retravail_min"]:
        df[col] = df[col].fillna(df[col].median())
    print("  ✓ Valeurs manquantes traitées")

    # Plages physiques
    masque_ok = pd.Series([True] * len(df), index=df.index)
    for col, (mini, maxi) in PLAGES_PHYSIQUES.items():
        masque_ok &= df[col].between(mini, maxi)
    nb_hors = (~masque_ok).sum()
    df = df[masque_ok].copy()
    print(f"  ✓ Plages physiques : {nb_hors} lignes hors norme supprimées")

    # Catégories
    for col, valides in VALEURS_VALIDES.items():
        invalides = ~df[col].isin(valides)
        df = df[~invalides].copy()
    print("  ✓ Catégories validées")

    # Cohérence métier
    df.loc[(df["statut_conformite"] == True) & (df["rebut"] == True), "rebut"] = False
    df.loc[(df["statut_conformite"] == True), "type_defaut"] = "conforme"
    df.loc[(df["retravail"] == False), "temps_retravail_min"] = 0.0
    print("  ✓ Cohérence métier vérifiée")

    nb_final = len(df)
    print(f"\n  Bilan : {nb_initial:,} lignes → {nb_final:,} conservées "
          f"({nb_initial - nb_final} supprimées)")

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════
# ÉTAPE 3 — LOAD
# ═══════════════════════════════════════════
def load(df: pd.DataFrame, engine) -> None:
    print("\n── LOAD ────────────────────────────────────")
    print(f"  Insertion de {len(df):,} lignes dans SQLite...")

    debut = datetime.now()
    df.to_sql(
        name="production_harness",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=CHUNK_SIZE,
        method="multi",
    )
    duree = (datetime.now() - debut).total_seconds()
    print(f"  ✓ Insertion terminée en {duree:.1f} secondes")


# ═══════════════════════════════════════════
# VÉRIFICATION POST-CHARGEMENT
# ═══════════════════════════════════════════
def verifier(engine) -> None:
    print("\n── VÉRIFICATION ────────────────────────────")

    with engine.connect() as conn:
        total    = conn.execute(text("SELECT COUNT(*) FROM production_harness")).fetchone()[0]
        conformes = conn.execute(text("SELECT COUNT(*) FROM production_harness WHERE statut_conformite = 1")).fetchone()[0]
        non_conf  = conn.execute(text("SELECT COUNT(*) FROM production_harness WHERE statut_conformite = 0")).fetchone()[0]
        date_min  = conn.execute(text("SELECT MIN(date_production) FROM production_harness")).fetchone()[0]
        date_max  = conn.execute(text("SELECT MAX(date_production) FROM production_harness")).fetchone()[0]
        nb_refs   = conn.execute(text("SELECT COUNT(DISTINCT reference) FROM production_harness")).fetchone()[0]
        nb_ops    = conn.execute(text("SELECT COUNT(DISTINCT operateur_id) FROM production_harness")).fetchone()[0]

    fpy = conformes / total * 100

    print(f"  Total faisceaux          : {total:,}")
    print(f"  Conformes                : {conformes:,}  ({fpy:.1f}%  ← FPY)")
    print(f"  Non-conformes            : {non_conf:,}  ({100-fpy:.1f}%)")
    print(f"  Période                  : {date_min[:10]} → {date_max[:10]}")
    print(f"  Références distinctes    : {nb_refs}")
    print(f"  Opérateurs distincts     : {nb_ops}")

    print("\n  Top défauts :")
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT type_defaut,
                   COUNT(*) AS nb,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM production_harness WHERE statut_conformite = 0), 1) AS pct
            FROM production_harness
            WHERE statut_conformite = 0
            GROUP BY type_defaut
            ORDER BY nb DESC
            LIMIT 5
        """)).fetchall()
    for r in rows:
        print(f"    {r[0]:<28} : {r[1]:>4} cas  ({r[2]}%)")

    print(f"\n  Fichier SQLite : data/sews_production.db")
    print("  ✓ Base prête pour la Phase 3 — KPI dynamiques !")
    print("=" * 55)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Pipeline ETL — SEWS Cabind  (SQLite)")
    print(f"  {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
    print("=" * 55)

    engine   = create_engine(get_url(), echo=False)
    df_brut  = extract(CSV_PATH)
    df_clean = transform(df_brut)
    load(df_clean, engine)
    verifier(engine)


if __name__ == "__main__":
    main()