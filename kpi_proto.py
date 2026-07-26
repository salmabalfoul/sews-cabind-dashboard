"""
KPI Prototype — Calcule les 5 KPI depuis la base SQLite réelle
Fichier : kpi_proto.py

KPI calculés :
  1. Présence et ponctualité
  2. Performance production
  3. Respect des temps standards
  4. Suivi de la coupe
  5. Primes (barème officiel SEWS)

Lancement : python kpi_proto.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os, warnings
warnings.filterwarnings("ignore")

DB_PATH = os.path.join("data", "sews_reel.db")

def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)

def charger(table, engine=None):
    if engine is None:
        engine = get_engine()
    return pd.read_sql(text(f"SELECT * FROM {table}"), engine)


# ═══════════════════════════════════════════════════════════
# KPI 1 — PRÉSENCE ET PONCTUALITÉ
# Source : tables employes + retards
# ═══════════════════════════════════════════════════════════
def kpi_presence():
    """
    Calcule pour chaque employé :
    - Nombre de jours de retard
    - Statut : Ponctuel / Quelques retards / Retards fréquents
    - Classement

    Données réelles : hayat=33, hiba=10, YADRI MALIKA=8,
    SOUR BOUCHRA=4, ZAROUAL=4, FATINE=2, ELATMANI=2,
    mana=1, KENZA=1, OUMNI=1, HIKLEF=1, ZAROUG=1
    """
    engine = get_engine()
    df_emp = charger("employes",    engine)
    df_ret = charger("retards",     engine)
    df_aff = charger("affectations",engine)

    # Nettoyage des noms pour jointure
    df_ret["nom_clean"] = df_ret["nom_prenom"].str.lower().str.strip()
    df_emp["nom_clean"] = df_emp["nom_prenom"].str.lower().str.strip()

    # Jointure sur nom nettoyé
    df = df_emp.merge(
        df_ret[["nom_clean","nb_jours_retard"]],
        on="nom_clean", how="left"
    ).fillna({"nb_jours_retard": 0})

    # Jointure avec zones d'affectation
    df_aff["nom_clean"] = df_aff["nom_prenom"].str.lower().str.strip()
    df = df.merge(
        df_aff[["nom_clean","zone"]].drop_duplicates("nom_clean"),
        on="nom_clean", how="left"
    )

    df["nb_jours_retard"] = df["nb_jours_retard"].astype(int)

    # Statut selon seuils
    def statut_retard(n):
        if n == 0:   return "🟢 Ponctuel"
        if n <= 3:   return "🟡 Quelques retards"
        if n <= 10:  return "🟠 Retards fréquents"
        return             "🔴 Retards critiques"

    df["statut_ponctualite"] = df["nb_jours_retard"].apply(statut_retard)

    # Taux de ponctualité global
    total_employes = len(df)
    nb_zero_retard = (df["nb_jours_retard"] == 0).sum()
    taux_ponctualite = round(nb_zero_retard / total_employes * 100, 1)

    print("\n═"*55)
    print("  KPI 1 — PRÉSENCE ET PONCTUALITÉ")
    print("═"*55)
    print(f"  Total employés     : {total_employes}")
    print(f"  Sans aucun retard  : {nb_zero_retard}  ({taux_ponctualite}%)")
    print(f"  Avec retards       : {total_employes - nb_zero_retard}")
    print(f"\n  Détail par employé :")
    print(f"  {'Nom':<30} {'Zone':<35} {'Retards':>8}  Statut")
    print(f"  {'-'*90}")
    for _, row in df.sort_values("nb_jours_retard", ascending=False).iterrows():
        zone = str(row.get("zone","?"))[:33]
        print(f"  {row['nom_prenom']:<30} {zone:<35} "
              f"{int(row['nb_jours_retard']):>8}  {row['statut_ponctualite']}")

    return df, {
        "total_employes":    total_employes,
        "nb_ponctuel":       int(nb_zero_retard),
        "taux_ponctualite":  taux_ponctualite,
        "total_jours_retard":int(df["nb_jours_retard"].sum()),
        "employe_plus_retard": df.loc[df["nb_jours_retard"].idxmax(), "nom_prenom"],
        "max_retards":       int(df["nb_jours_retard"].max()),
    }


# ═══════════════════════════════════════════════════════════
# KPI 2 — PERFORMANCE PRODUCTION
# Source : tables affectations + temps_standards
# ═══════════════════════════════════════════════════════════
def kpi_performance():
    """
    Calcule la performance par zone de production.

    Données réelles disponibles :
    - CABINA    : ouala + Idrissi (tps std = 7.5h/pièce)
    - Engine    : saadeddine + mana (tps std = 4h/pièce)
    - BRIGLIA   : LOUZ MOUNA (tps std = 2h/pièce)
    - COFANO    : 7 opérateurs (tps std = 15h/pièce)
    - CONTRÔLE  : hiba + JAOUDALLAH + MOUNA ZAIR
    - PREMONTAGE: HMIDCHAT + OUMNI + alif + babali
    - SERTISSAGE: BADREZZAMANE + hayat
    """
    engine = get_engine()
    df_aff = charger("affectations",   engine)
    df_tps = charger("temps_standards",engine)

    # Mapping zones → familles pour les temps standards
    zone_famille = {
        "CABINA":                              "CABINA",
        "Engine":                              "Engine",
        "BRIGLIA UREA":                        "BRIGLIA UREA",
        "COFANO":                              "COFANO",
        "COFANO NDE":                          "COFANO",
        "CONTRÔLE ELECTRIQUE+PIN TO PIN":      "ALL",
        "CONTRÔLE FINAL":                      "ALL",
        "PREMONTAGE":                          "ALL",
        "PREPARATION GAINE":                   "ALL",
        "épissurage":                          "ALL",
        "SERTISSAGE +épissurage":              "ALL",
        "support PM":                          "ALL",
        "PREPARATION ET VALIDATION LES TABLES DE MONTAGE": "ALL",
    }

    df_aff["famille"] = df_aff["zone"].map(zone_famille).fillna("ALL")

    # Temps standards par famille
    tps_dict = dict(zip(df_tps["famille"], df_tps["tps_proto_h"]))

    # Résumé par zone
    zones = df_aff.groupby("zone").agg(
        nb_operateurs=("nom_prenom","count"),
        famille=("famille","first"),
    ).reset_index()

    zones["tps_standard_h"] = zones["famille"].map(tps_dict)
    zones["cadence_ref"]     = zones["famille"].map(
        dict(zip(df_tps["famille"], df_tps["cadence_txt"]))
    )

    print("\n═"*55)
    print("  KPI 2 — PERFORMANCE PRODUCTION")
    print("═"*55)
    print(f"\n  {'Zone':<40} {'Nb Op':>6} {'Tps Std (h)':>12} {'Cadence':>20}")
    print(f"  {'-'*80}")
    for _, row in zones.iterrows():
        tps = f"{row['tps_standard_h']:.2f}h" if pd.notna(row["tps_standard_h"]) else "variable"
        cad = str(row["cadence_ref"]) if pd.notna(row["cadence_ref"]) else "—"
        print(f"  {row['zone'][:38]:<40} {int(row['nb_operateurs']):>6} "
              f"{tps:>12} {cad:>20}")

    print(f"\n  Opérateurs par zone :")
    for zone, grp in df_aff.groupby("zone"):
        noms = ", ".join(grp["nom_prenom"].tolist())
        print(f"  {zone[:38]:<40}: {noms}")

    return df_aff, zones


# ═══════════════════════════════════════════════════════════
# KPI 3 — RESPECT DES TEMPS STANDARDS (Manhours)
# Source : table manhours_2025 + temps_standards
# ═══════════════════════════════════════════════════════════
def kpi_temps_standards():
    """
    Compare les manhours consommées vs théoriques.

    Exemple réel depuis Classeur3.xlsx :
    Briglia UREA Janvier 2025 : 31 pièces × 2h/pièce = 62h théoriques
    vs 43.4h réelles → efficacité = 62/43.4 = 142% (très bon !)
    """
    engine = get_engine()
    df_mh  = charger("manhours_2025",  engine)
    df_tps = charger("temps_standards",engine)

    if df_mh is None or len(df_mh) == 0:
        print("  ⚠ Données manhours non disponibles")
        return None, {}

    # Calcul efficacité
    df_mh["mh_theoriques"] = df_mh["quantite"] * df_mh["tps_proto_h"].fillna(0)
    df_mh["efficacite_pct"] = np.where(
        df_mh["manhours"] > 0,
        (df_mh["mh_theoriques"] / df_mh["manhours"] * 100).round(1),
        0
    )
    df_mh["ecart_mh"] = (df_mh["manhours"] - df_mh["mh_theoriques"]).round(1)

    # Résumé par sous-projet
    resume = df_mh.groupby("sous_projet").agg(
        qte_totale   =("quantite",    "sum"),
        mh_reelles   =("manhours",    "sum"),
        mh_theo      =("mh_theoriques","sum"),
    ).reset_index()
    resume["efficacite_pct"] = (resume["mh_theo"] / resume["mh_reelles"] * 100).round(1)
    resume["ecart_mh"]       = (resume["mh_reelles"] - resume["mh_theo"]).round(1)

    print("\n═"*55)
    print("  KPI 3 — RESPECT DES TEMPS STANDARDS")
    print("═"*55)
    print(f"\n  {'Projet':<20} {'QTE':>6} {'MH Réelles':>12} "
          f"{'MH Théo':>10} {'Efficacité':>11} {'Écart MH':>10}")
    print(f"  {'-'*73}")
    for _, row in resume.sort_values("efficacite_pct", ascending=False).iterrows():
        statut = "✓" if row["efficacite_pct"] >= 100 else "⚠"
        print(f"  {row['sous_projet']:<20} {row['qte_totale']:>6.0f} "
              f"{row['mh_reelles']:>12.1f} {row['mh_theo']:>10.1f} "
              f"{row['efficacite_pct']:>10.1f}% {row['ecart_mh']:>9.1f}h  {statut}")

    eff_glob = (resume["mh_theo"].sum() / resume["mh_reelles"].sum() * 100)
    print(f"\n  Efficacité globale : {eff_glob:.1f}%")
    print(f"  MH réelles totales : {resume['mh_reelles'].sum():.0f}h")
    print(f"  MH théoriques tot. : {resume['mh_theo'].sum():.0f}h")

    return df_mh, {
        "efficacite_globale": round(eff_glob, 1),
        "mh_reelles_total":   round(resume["mh_reelles"].sum(), 1),
        "mh_theoriques_total":round(resume["mh_theo"].sum(), 1),
        "ecart_total":        round(resume["ecart_mh"].sum(), 1),
    }


# ═══════════════════════════════════════════════════════════
# KPI 4 — SUIVI DE LA COUPE
# Source : tables suivi_coupe + ordres_komax
# ═══════════════════════════════════════════════════════════
def kpi_coupe():
    """
    Suivi de la coupe WK24 :
    - Cabina      : 82.4% coupé (122/148) ✓
    - Briglia UREA: 67.2% coupé (45/67)   →
    - COFANO 02BB : 44.8% coupé (218/487) ⚠
    - COFANO 00A1 : 0.0%  coupé (0/486)   🔴 CRITIQUE

    702 ordres KOMAX : COFANO=477, CABINA=148, BRIGLIA=67, GS=10
    """
    engine   = get_engine()
    df_coupe = charger("suivi_coupe",  engine)
    df_komax = charger("ordres_komax", engine)

    # Statut par faisceau
    def statut_coupe(pct):
        if pct >= 80:  return "🟢 OK"
        if pct >= 50:  return "🟡 En cours"
        if pct > 0:    return "🟠 En retard"
        return               "🔴 BLOQUÉ"

    df_coupe["statut"] = df_coupe["pct_coupe"].apply(statut_coupe)

    print("\n═"*55)
    print("  KPI 4 — SUIVI DE LA COUPE WK24")
    print("═"*55)
    print(f"\n  {'Famille':<22} {'Réf':<22} {'Coupé':>7} "
          f"{'Total':>7} {'%':>7}  Statut")
    print(f"  {'-'*78}")
    for _, row in df_coupe.iterrows():
        print(f"  {row['famille'][:20]:<22} {row['reference'][:20]:<22} "
              f"{row['coupe']:>7} {row['nb_reperes']:>7} "
              f"{row['pct_coupe']:>6.1f}%  {row['statut']}")

    # Stats KOMAX
    komax_stats = df_komax["Description"].value_counts()
    komax_locked  = df_komax["est_locked"].sum()
    komax_termine = df_komax["est_termine"].sum()

    print(f"\n  Ordres KOMAX :")
    for desc, n in komax_stats.items():
        print(f"    {desc:<20} : {n} ordres")
    print(f"  Ordres locked (planifiés) : {komax_locked}")
    print(f"  Ordres terminés           : {komax_termine}")
    print(f"\n  ⚠ ALERTE : COFANO PR5803621806 00A1 = 0% coupé !")
    print(f"    Indice de lancement = '?????' — À clarifier urgemment")

    return df_coupe, df_komax, {
        "nb_faisceaux_suivi": len(df_coupe),
        "nb_critique":        int((df_coupe["pct_coupe"] == 0).sum()),
        "nb_retard":          int((df_coupe["pct_coupe"].between(1, 49)).sum()),
        "nb_ok":              int((df_coupe["pct_coupe"] >= 80).sum()),
        "total_ordres_komax": len(df_komax),
        "ordres_locked":      int(komax_locked),
    }


# ═══════════════════════════════════════════════════════════
# KPI 5 — CALCUL DES PRIMES
# Barème officiel SEWS (image fournie)
# ═══════════════════════════════════════════════════════════
def kpi_primes(df_employes_retards=None):
    """
    Barème officiel des primes (image fournie) :
    Objectif+2%  → 50 MAD    | Avec 0 abs 3 mois → 60 MAD
    +4%  → 100 MAD  | 120 MAD
    +6%  → 150 MAD  | 180 MAD
    +8%  → 200 MAD  | 240 MAD
    +10% → 250 MAD  | 300 MAD
    +12% → 300 MAD  | 360 MAD
    +14% → 350 MAD  | 420 MAD
    +16% → 400 MAD  | 480 MAD
    +18% → 450 MAD  | 540 MAD
    +20% → 500 MAD  | 600 MAD

    La prime est divisée : 50% Productivité + 50% Qualité
    """
    BAREME = {
        2:  (25,  25),    # (prime_prod, prime_qualite) en MAD
        4:  (50,  50),
        6:  (75,  75),
        8:  (100, 100),
        10: (125, 125),
        12: (150, 150),
        14: (175, 175),
        16: (200, 200),
        18: (225, 225),
        20: (250, 250),
    }

    def calculer_prime(taux_amelioration, zero_absence_3mois=False):
        """Calcule la prime selon le taux d'amélioration et l'assiduité."""
        prime_prod = prime_qual = 0
        for seuil in sorted(BAREME.keys(), reverse=True):
            if taux_amelioration >= seuil:
                prime_prod, prime_qual = BAREME[seuil]
                break
        total = prime_prod + prime_qual
        if zero_absence_3mois and total > 0:
            total = round(total * 1.20)  # +20% bonus assiduité
        return prime_prod, prime_qual, total

    engine = get_engine()
    df_ret = charger("retards",      engine)
    df_aff = charger("affectations", engine)
    df_emp = charger("employes",     engine)

    # Pour la démonstration, on simule des taux d'amélioration
    # basés sur les données disponibles
    # En réalité, ce taux vient de : performance actuelle vs période précédente
    np.random.seed(42)
    taux_simules = {
        "ouala":            18,   # bon opérateur CABINA
        "Idrissi Bouchra":  14,
        "saadeddine aziza": 16,
        "mana":             10,
        "LOUZ MOUNA":       20,   # 100% performance réelle
        "HMIDCHAT AICHA":   18,   # 100% performance réelle
        "hiba":             2,    # 53.8% perf → amélioration limitée
        "hayat":            2,    # 33 retards → prime minimale
        "YADRI MALIKA":     4,    # 8 retards
        "SOUR BOUCHRA":     6,
        "ANIBA NEZHA":      6,
        "erradah":          8,
        "Soukaina IHIKIN":  8,
        "FATINE BOUABDELY": 4,
    }

    rows_prime = []
    for _, emp in df_emp.iterrows():
        nom = emp["nom_prenom"].strip()
        nom_clean = nom.lower().strip()

        # Trouver le taux pour cet employé
        taux = 2  # défaut minimal
        for k, v in taux_simules.items():
            if k.lower() in nom_clean or nom_clean in k.lower():
                taux = v
                break

        # 0 absence = retard == 0
        ret_row = df_ret[df_ret["nom_prenom"].str.lower().str.strip() == nom_clean]
        nb_retards = int(ret_row["nb_jours_retard"].values[0]) if len(ret_row) > 0 else 0
        zero_absence = (nb_retards == 0)

        pp, pq, total = calculer_prime(taux, zero_absence)

        rows_prime.append({
            "nom_prenom":       nom,
            "taux_amelioration":taux,
            "prime_productivite":pp,
            "prime_qualite":    pq,
            "prime_base":       pp + pq,
            "bonus_assiduite":  round(total - (pp+pq)),
            "prime_totale":     total,
            "zero_retard":      zero_absence,
            "nb_retards":       nb_retards,
        })

    df_primes = pd.DataFrame(rows_prime).sort_values(
        "prime_totale", ascending=False
    )

    print("\n═"*55)
    print("  KPI 5 — PRIMES DU MOIS")
    print("═"*55)
    print(f"\n  {'Nom':<30} {'Taux %':>7} {'P.Prod':>8} "
          f"{'P.Qual':>8} {'Bonus':>7} {'TOTAL MAD':>10}  Retards")
    print(f"  {'-'*85}")
    for _, row in df_primes.iterrows():
        print(f"  {row['nom_prenom'][:28]:<30} "
              f"{row['taux_amelioration']:>6}% "
              f"{row['prime_productivite']:>8} "
              f"{row['prime_qualite']:>8} "
              f"{row['bonus_assiduite']:>7} "
              f"{row['prime_totale']:>10} MAD  "
              f"{row['nb_retards']} jr")

    total_a_payer = df_primes["prime_totale"].sum()
    print(f"\n  TOTAL PRIMES À PAYER : {total_a_payer:,} MAD")
    print(f"  Employés avec prime  : {(df_primes['prime_totale']>0).sum()}")
    print(f"  Prime max            : {df_primes['prime_totale'].max()} MAD "
          f"({df_primes.loc[df_primes['prime_totale'].idxmax(),'nom_prenom']})")

    return df_primes, {
        "total_primes_mad":  int(total_a_payer),
        "nb_avec_prime":     int((df_primes["prime_totale"]>0).sum()),
        "prime_max":         int(df_primes["prime_totale"].max()),
        "prime_min":         int(df_primes[df_primes["prime_totale"]>0]["prime_totale"].min()),
    }


# ═══════════════════════════════════════════════════════════
# RAPPORT COMPLET
# ═══════════════════════════════════════════════════════════
def rapport_complet():
    """Lance tous les KPI et retourne un résumé."""
    print("=" * 55)
    print("  Rapport KPI Prototype — SEWS Cabind")
    print("=" * 55)

    df_pres, stats_pres = kpi_presence()
    df_aff,  zones      = kpi_performance()
    df_mh,   stats_mh   = kpi_temps_standards()
    df_coupe, df_kx, stats_coupe = kpi_coupe()
    df_primes, stats_primes      = kpi_primes()

    print("\n" + "=" * 55)
    print("  RÉSUMÉ EXÉCUTIF")
    print("=" * 55)
    print(f"  Ponctualité      : {stats_pres['taux_ponctualite']}% sans retard")
    print(f"  Retard max       : {stats_pres['max_retards']} jours "
          f"({stats_pres['employe_plus_retard']})")
    print(f"  Efficacité MH    : {stats_mh.get('efficacite_globale','N/A')}%")
    print(f"  Faisceaux bloqués: {stats_coupe['nb_critique']} (0% coupé)")
    print(f"  Primes à payer   : {stats_primes['total_primes_mad']:,} MAD")
    print("=" * 55)

    return {
        "presences": (df_pres, stats_pres),
        "performance": (df_aff, zones),
        "manhours": (df_mh, stats_mh),
        "coupe": (df_coupe, df_kx, stats_coupe),
        "primes": (df_primes, stats_primes),
    }


if __name__ == "__main__":
    rapport_complet()