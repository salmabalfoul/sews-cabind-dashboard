"""
Phase 3 — Moteur de calcul des KPI dynamiques
Projet : Système intelligent de pilotage des faisceaux — SEWS Cabind

Ce fichier calcule tous les KPI depuis la base SQLite.
Il peut être appelé depuis le dashboard (Phase 5) ou en standalone.

KPI calculés :
  1. FPY          — First Pass Yield
  2. PPM          — Parts Per Million
  3. Taux rebut   — Proportion de faisceaux irrécupérables
  4. Pareto       — Classement des défauts par fréquence
  5. Tendances    — Évolution hebdomadaire des KPI
  + Comparaisons  — Par shift, opérateur, référence, ligne
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from db_config import get_url

# ─────────────────────────────────────────
# CONNEXION À LA BASE
# ─────────────────────────────────────────
def get_engine():
    return create_engine(get_url(), echo=False)


def charger_donnees(
    date_debut: str = None,
    date_fin:   str = None,
    reference:  str = None,
    ligne:      str = None,
    shift:      str = None,
) -> pd.DataFrame:
    """
    Charge les données depuis SQLite avec des filtres optionnels.

    Paramètres
    ----------
    date_debut : str  ex: '2023-01-01'
    date_fin   : str  ex: '2023-12-31'
    reference  : str  ex: 'BDX-0047'  (None = toutes)
    ligne      : str  ex: 'Ligne_A'   (None = toutes)
    shift      : str  ex: 'nuit'      (None = tous)

    Retourne
    --------
    DataFrame avec toutes les colonnes de production_harness
    """
    engine = get_engine()

    # Construction de la requête avec filtres dynamiques
    conditions = []
    params     = {}

    if date_debut:
        conditions.append("date_production >= :date_debut")
        params["date_debut"] = date_debut

    if date_fin:
        conditions.append("date_production <= :date_fin")
        params["date_fin"] = date_fin + " 23:59:59"

    if reference and reference != "Toutes":
        conditions.append("reference = :reference")
        params["reference"] = reference.lower()

    if ligne and ligne != "Toutes":
        conditions.append("ligne_production = :ligne")
        params["ligne"] = ligne.lower()

    if shift and shift != "Tous":
        conditions.append("shift = :shift")
        params["shift"] = shift.lower()

    clause_where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM production_harness {clause_where}"

    df = pd.read_sql(text(sql), engine, params=params)
    df["date_production"] = pd.to_datetime(df["date_production"])
    return df


# ═══════════════════════════════════════════
# KPI 1 — FIRST PASS YIELD (FPY)
# ═══════════════════════════════════════════
def calculer_fpy(df: pd.DataFrame) -> dict:
    """
    First Pass Yield = faisceaux conformes sans retravail / total × 100

    Le FPY est plus strict que le simple taux de conformité :
    il exclut les pièces qui ont nécessité une retouche,
    même si elles sont finalement conformes.

    Retourne un dictionnaire avec :
    - fpy          : valeur en pourcentage (ex: 73.2)
    - conformes    : nombre de pièces conformes du 1er coup
    - total        : total de pièces
    - statut       : 'bon' / 'attention' / 'critique'
    - benchmark    : objectif industrie
    """
    if len(df) == 0:
        return {"fpy": 0, "conformes": 0, "total": 0,
                "statut": "critique", "benchmark": ">95% en série"}

    total     = len(df)
    # Conformes du premier coup = conforme ET pas de retravail
    conformes_1er_coup = df[
        (df["statut_conformite"] == 1) & (df["retravail"] == 0)
    ].shape[0]

    fpy = round(conformes_1er_coup / total * 100, 2)

    # Évaluation par rapport aux benchmarks industrie
    if fpy >= 95:
        statut = "bon"        # objectif série atteint
    elif fpy >= 80:
        statut = "attention"  # acceptable en prototype
    else:
        statut = "critique"   # nécessite une action

    return {
        "fpy":       fpy,
        "conformes": conformes_1er_coup,
        "total":     total,
        "statut":    statut,
        "benchmark": ">95% en série / >80% en prototype",
    }


# ═══════════════════════════════════════════
# KPI 2 — PPM (Parts Per Million)
# ═══════════════════════════════════════════
def calculer_ppm(df: pd.DataFrame) -> dict:
    """
    PPM = (nombre de défauts / total produit) × 1 000 000

    Unité standard dans l'industrie automobile.
    Permet de comparer des volumes de production différents.

    Retourne :
    - ppm        : valeur PPM
    - nb_defauts : nombre absolu de défauts
    - total      : total produit
    - statut     : évaluation qualité
    """
    if len(df) == 0:
        return {"ppm": 0, "nb_defauts": 0, "total": 0, "statut": "inconnu"}

    total      = len(df)
    nb_defauts = df[df["statut_conformite"] == 0].shape[0]
    ppm        = round(nb_defauts / total * 1_000_000, 0)

    if ppm < 500:
        statut = "bon"         # objectif série
    elif ppm < 5_000:
        statut = "attention"   # acceptable en développement
    elif ppm < 50_000:
        statut = "critique"
    else:
        statut = "tres_critique"  # prototype initial normal

    return {
        "ppm":        int(ppm),
        "nb_defauts": nb_defauts,
        "total":      total,
        "statut":     statut,
        "benchmark":  "<500 PPM en série / <50 000 en prototype",
    }


# ═══════════════════════════════════════════
# KPI 3 — TAUX DE REBUT ET RETRAVAIL
# ═══════════════════════════════════════════
def calculer_rebut_retravail(df: pd.DataFrame) -> dict:
    """
    Taux de rebut    = pièces irrécupérables / total × 100
    Taux de retravail = pièces retouchées / total × 100
    Coût retravail   = somme des temps de retouche (minutes)

    Les pièces au rebut sont perdues (matière + main d'œuvre).
    Les pièces en retravail sont récupérées mais coûtent du temps.
    """
    if len(df) == 0:
        return {
            "taux_rebut": 0, "nb_rebut": 0,
            "taux_retravail": 0, "nb_retravail": 0,
            "temps_retravail_total_h": 0,
            "temps_retravail_moyen_min": 0,
        }

    total         = len(df)
    nb_rebut      = df[df["rebut"] == 1].shape[0]
    nb_retravail  = df[df["retravail"] == 1].shape[0]

    taux_rebut     = round(nb_rebut / total * 100, 2)
    taux_retravail = round(nb_retravail / total * 100, 2)

    # Temps total de retouche (converti en heures)
    temps_total_min  = df[df["retravail"] == 1]["temps_retravail_min"].sum()
    temps_moyen_min  = df[df["retravail"] == 1]["temps_retravail_min"].mean()

    return {
        "taux_rebut":               taux_rebut,
        "nb_rebut":                 nb_rebut,
        "taux_retravail":           taux_retravail,
        "nb_retravail":             nb_retravail,
        "temps_retravail_total_h":  round(temps_total_min / 60, 1),
        "temps_retravail_moyen_min": round(temps_moyen_min, 1) if nb_retravail > 0 else 0,
        "benchmark_rebut":          "<2% en série",
        "benchmark_retravail":      "<5% en série",
    }


# ═══════════════════════════════════════════
# KPI 4 — ANALYSE DE PARETO
# ═══════════════════════════════════════════
def calculer_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classement des défauts par fréquence décroissante.
    Inclut le pourcentage cumulé (règle des 80/20).

    La règle de Pareto dit que 20% des causes provoquent
    80% des problèmes. Le Pareto identifie ces 20%.

    Retourne un DataFrame avec :
    - type_defaut  : nom du défaut
    - nb           : nombre d'occurrences
    - pct          : pourcentage du total des défauts
    - pct_cumule   : pourcentage cumulé (pour la courbe Pareto)
    - priorite     : 'haute' si dans les 80% cumulés
    """
    # On ne garde que les non-conformes
    df_defauts = df[df["statut_conformite"] == 0].copy()

    if len(df_defauts) == 0:
        return pd.DataFrame(columns=[
            "type_defaut", "nb", "pct", "pct_cumule", "priorite"
        ])

    # Comptage par type de défaut
    pareto = (
        df_defauts["type_defaut"]
        .value_counts()
        .reset_index()
    )
    pareto.columns = ["type_defaut", "nb"]

    # Calcul des pourcentages
    total_defauts  = pareto["nb"].sum()
    pareto["pct"]  = (pareto["nb"] / total_defauts * 100).round(2)

    # Pourcentage cumulé (c'est la courbe en S du Pareto)
    pareto["pct_cumule"] = pareto["pct"].cumsum().round(2)

    # Marquage de priorité : dans les 80% cumulés = priorité haute
    pareto["priorite"] = pareto["pct_cumule"].apply(
        lambda x: "haute" if x <= 80 else "normale"
    )

    return pareto


# ═══════════════════════════════════════════
# KPI 5 — TENDANCES HEBDOMADAIRES
# ═══════════════════════════════════════════
def calculer_tendances(df: pd.DataFrame, nb_semaines: int = 8) -> pd.DataFrame:
    """
    Calcule le FPY et le taux de défauts semaine par semaine
    sur les nb_semaines dernières semaines.

    Permet de répondre à : "La qualité s'améliore-t-elle ?"

    Retourne un DataFrame avec une ligne par semaine :
    - semaine      : numéro de semaine (ex: '2024-W12')
    - total        : faisceaux produits cette semaine
    - fpy          : FPY de la semaine
    - ppm          : PPM de la semaine
    - taux_defaut  : taux de non-conformité
    - tendance_fpy : variation vs semaine précédente (+/-/=)
    """
    if len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df["semaine"] = df["date_production"].dt.to_period("W").astype(str)

    # Calcul des métriques par semaine
    def metriques_semaine(g):
        total     = len(g)
        conformes = ((g["statut_conformite"] == 1) & (g["retravail"] == 0)).sum()
        defauts   = (g["statut_conformite"] == 0).sum()
        fpy       = round(conformes / total * 100, 2) if total > 0 else 0
        ppm       = round(defauts / total * 1_000_000, 0) if total > 0 else 0
        taux_def  = round(defauts / total * 100, 2) if total > 0 else 0
        return pd.Series({
            "total":       total,
            "fpy":         fpy,
            "ppm":         int(ppm),
            "taux_defaut": taux_def,
        })

    tendances = (
        df.groupby("semaine")
        .apply(metriques_semaine)
        .reset_index()
        .sort_values("semaine")
    )

    # Calcul de la tendance FPY (variation vs semaine précédente)
    tendances["variation_fpy"] = tendances["fpy"].diff().round(2)
    tendances["tendance"] = tendances["variation_fpy"].apply(
        lambda x: "hausse" if x > 0.5
        else ("baisse" if x < -0.5 else "stable")
        if pd.notna(x) else "—"
    )

    # Retourne uniquement les nb_semaines dernières semaines
    return tendances.tail(nb_semaines).reset_index(drop=True)


# ═══════════════════════════════════════════
# COMPARAISONS MULTI-DIMENSIONS
# ═══════════════════════════════════════════
def comparer_par_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare le FPY et le taux de défauts par shift.
    Répond à : 'Le shift de nuit produit-il plus de défauts ?'
    """
    if len(df) == 0:
        return pd.DataFrame()

    def metriques(g):
        total     = len(g)
        conformes = ((g["statut_conformite"] == 1) & (g["retravail"] == 0)).sum()
        defauts   = (g["statut_conformite"] == 0).sum()
        rebuts    = (g["rebut"] == 1).sum()
        return pd.Series({
            "total":       total,
            "fpy":         round(conformes / total * 100, 2),
            "taux_defaut": round(defauts / total * 100, 2),
            "taux_rebut":  round(rebuts / total * 100, 2),
            "ppm":         int(defauts / total * 1_000_000),
        })

    return (
        df.groupby("shift")
        .apply(metriques)
        .reset_index()
        .sort_values("fpy", ascending=False)
    )


def comparer_par_operateur(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compare le FPY par opérateur.
    Répond à : 'Quels opérateurs ont besoin de formation ?'
    Retourne les top_n meilleurs et les top_n moins bons.
    """
    if len(df) == 0:
        return pd.DataFrame()

    def metriques(g):
        total     = len(g)
        if total < 10:  # Pas assez de données pour être significatif
            return None
        conformes = ((g["statut_conformite"] == 1) & (g["retravail"] == 0)).sum()
        defauts   = (g["statut_conformite"] == 0).sum()
        return pd.Series({
            "total":       total,
            "fpy":         round(conformes / total * 100, 2),
            "taux_defaut": round(defauts / total * 100, 2),
            "nb_defauts":  defauts,
        })

    result = (
        df.groupby("operateur_id")
        .apply(metriques)
        .dropna()
        .reset_index()
        .sort_values("fpy", ascending=False)
    )
    return result


def comparer_par_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare la qualité par référence de faisceau.
    Répond à : 'Quelle référence est la plus difficile à produire ?'
    """
    if len(df) == 0:
        return pd.DataFrame()

    def metriques(g):
        total     = len(g)
        conformes = ((g["statut_conformite"] == 1) & (g["retravail"] == 0)).sum()
        defauts   = (g["statut_conformite"] == 0).sum()
        temps_moy = g["temps_cycle_min"].mean()
        return pd.Series({
            "total":           total,
            "fpy":             round(conformes / total * 100, 2),
            "taux_defaut":     round(defauts / total * 100, 2),
            "ppm":             int(defauts / total * 1_000_000),
            "temps_cycle_moy": round(temps_moy, 1),
        })

    return (
        df.groupby("reference")
        .apply(metriques)
        .reset_index()
        .sort_values("fpy", ascending=False)
    )


def comparer_par_ligne(df: pd.DataFrame) -> pd.DataFrame:
    """Compare la performance par ligne de production."""
    if len(df) == 0:
        return pd.DataFrame()

    def metriques(g):
        total     = len(g)
        conformes = ((g["statut_conformite"] == 1) & (g["retravail"] == 0)).sum()
        defauts   = (g["statut_conformite"] == 0).sum()
        return pd.Series({
            "total":       total,
            "fpy":         round(conformes / total * 100, 2),
            "taux_defaut": round(defauts / total * 100, 2),
            "ppm":         int(defauts / total * 1_000_000),
        })

    return (
        df.groupby("ligne_production")
        .apply(metriques)
        .reset_index()
        .sort_values("fpy", ascending=False)
    )


# ═══════════════════════════════════════════
# RAPPORT COMPLET — toutes les métriques
# ═══════════════════════════════════════════
def rapport_complet(
    date_debut: str = None,
    date_fin:   str = None,
    reference:  str = None,
    ligne:      str = None,
    shift:      str = None,
) -> dict:
    """
    Calcule tous les KPI en une seule fois.
    C'est la fonction principale appelée par le dashboard.

    Retourne un dictionnaire avec toutes les métriques.
    """
    df = charger_donnees(date_debut, date_fin, reference, ligne, shift)

    if len(df) == 0:
        return {"erreur": "Aucune donnée pour ces filtres"}

    return {
        "nb_faisceaux":   len(df),
        "periode":        {
            "debut": str(df["date_production"].min().date()),
            "fin":   str(df["date_production"].max().date()),
        },
        "fpy":            calculer_fpy(df),
        "ppm":            calculer_ppm(df),
        "rebut_retravail": calculer_rebut_retravail(df),
        "pareto":         calculer_pareto(df),
        "tendances":      calculer_tendances(df),
        "par_shift":      comparer_par_shift(df),
        "par_reference":  comparer_par_reference(df),
        "par_ligne":      comparer_par_ligne(df),
        "par_operateur":  comparer_par_operateur(df),
    }


# ═══════════════════════════════════════════
# AFFICHAGE CONSOLE — pour tester en standalone
# ═══════════════════════════════════════════
def afficher_rapport(rapport: dict) -> None:
    """Affiche un rapport lisible dans le terminal."""

    print("=" * 60)
    print("  RAPPORT KPI — SEWS Cabind")
    print(f"  Période : {rapport['periode']['debut']} → {rapport['periode']['fin']}")
    print(f"  Total faisceaux analysés : {rapport['nb_faisceaux']:,}")
    print("=" * 60)

    # ── FPY ──
    fpy = rapport["fpy"]
    symbole = "✓" if fpy["statut"] == "bon" else ("!" if fpy["statut"] == "attention" else "✗")
    print(f"\n  [{symbole}] FPY (First Pass Yield)  : {fpy['fpy']}%")
    print(f"      Conformes 1er coup     : {fpy['conformes']:,} / {fpy['total']:,}")
    print(f"      Benchmark              : {fpy['benchmark']}")

    # ── PPM ──
    ppm = rapport["ppm"]
    print(f"\n  PPM                        : {ppm['ppm']:,}")
    print(f"      Défauts détectés       : {ppm['nb_defauts']:,}")
    print(f"      Benchmark              : {ppm['benchmark']}")

    # ── Rebut / Retravail ──
    rr = rapport["rebut_retravail"]
    print(f"\n  Taux de rebut              : {rr['taux_rebut']}%  ({rr['nb_rebut']} pièces)")
    print(f"  Taux de retravail          : {rr['taux_retravail']}%  ({rr['nb_retravail']} pièces)")
    print(f"  Temps retravail total      : {rr['temps_retravail_total_h']} heures")

    # ── Pareto ──
    pareto = rapport["pareto"]
    print(f"\n  ANALYSE DE PARETO (top défauts)")
    print(f"  {'Défaut':<28} {'Nb':>5}  {'%':>6}  {'% cumulé':>9}  Priorité")
    print(f"  {'-'*60}")
    for _, row in pareto.iterrows():
        print(f"  {row['type_defaut']:<28} {int(row['nb']):>5}  {row['pct']:>5.1f}%  "
              f"{row['pct_cumule']:>8.1f}%  {row['priorite']}")

    # ── Par shift ──
    print(f"\n  COMPARAISON PAR SHIFT")
    print(f"  {'Shift':<15} {'Total':>6}  {'FPY':>7}  {'Défauts':>8}  {'PPM':>8}")
    print(f"  {'-'*52}")
    for _, row in rapport["par_shift"].iterrows():
        print(f"  {row['shift']:<15} {int(row['total']):>6}  "
              f"{row['fpy']:>6.1f}%  {row['taux_defaut']:>7.1f}%  {row['ppm']:>8,}")

    # ── Tendances (4 dernières semaines) ──
    tendances = rapport["tendances"].tail(4)
    print(f"\n  TENDANCES (4 dernières semaines)")
    print(f"  {'Semaine':<14} {'Total':>6}  {'FPY':>7}  {'Tendance':>10}")
    print(f"  {'-'*44}")
    for _, row in tendances.iterrows():
        fleche = "↑" if row["tendance"] == "hausse" else ("↓" if row["tendance"] == "baisse" else "→")
        print(f"  {row['semaine']:<14} {int(row['total']):>6}  "
              f"{row['fpy']:>6.1f}%  {fleche} {row['tendance']}")

    print("\n" + "=" * 60)
    print("  ✓ Rapport terminé — Phase 3 complète")
    print("=" * 60)


# ═══════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("\nCalcul du rapport KPI complet...")
    print("(toutes les données, aucun filtre)\n")

    rapport = rapport_complet()
    afficher_rapport(rapport)

    # Exemple avec filtre : seulement le shift de nuit en 2024
    print("\n\nExemple filtré : shift de nuit, année 2024")
    rapport_nuit = rapport_complet(
        date_debut="2024-01-01",
        date_fin="2024-12-31",
        shift="nuit"
    )
    afficher_rapport(rapport_nuit)