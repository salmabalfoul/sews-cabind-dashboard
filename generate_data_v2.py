"""
Partie 1 — Version mise à jour avec données SEWS Cabind réelles
Fichier : generate_data_v2.py

CHANGEMENTS vs l'ancienne version :
  - Vraies références de faisceaux SEWS (COFANO, CABINA, ENGINE...)
  - Vrais prénoms des opérateurs de l'équipe proto
  - Vrais temps standards (Temps_standard_PROTO.xlsx)
  - Un seul shift "journee" (SEWS proto = pas de shift nuit)
  - Défauts cohérents avec le contexte faisceau automobile IVECO
  - FPY cohérent avec les performances réelles observées (~75-85%)

Lancement : python generate_data_v2.py
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
SEED         = 42
np.random.seed(SEED)
random.seed(SEED)

NB_FAISCEAUX = 5000           # réduit car proto = volumes plus faibles
DATE_DEBUT   = datetime(2024, 1, 1)
DATE_FIN     = datetime(2025, 12, 31)
OUTPUT_DIR   = "data"
OUTPUT_FILE  = "production_harness_sews.csv"

# ─────────────────────────────────────────────────────────
# VRAIES RÉFÉRENCES SEWS (depuis tes fichiers réels)
# ─────────────────────────────────────────────────────────
FAMILLES_FAISCEAUX = {
    # Famille : (référence, nb_circuits_min, nb_circuits_max, temps_standard_h)
    "COFANO":       ("PR5803621787", 60, 85, 5.76),
    "CABINA":       ("PR5803561063", 30, 50, 2.34),
    "BRIGLIA_UREA": ("PR5803607573", 15, 25, 2.00),
    "ENGINE":       ("PR5803686850", 10, 18, 1.06),
    "PDB":          ("PR5803549592", 40, 60, 7.54),
    "TELAIO":       ("PR5803495567", 35, 55, 4.84),
}

# ─────────────────────────────────────────────────────────
# VRAIS PRÉNOMS DE L'ÉQUIPE PROTO SEWS
# (depuis Employee_retard_schedule.xlsx feuille team)
# ─────────────────────────────────────────────────────────
OPERATEURS = {
    # Prénom    : (zone_affectation, niveau_performance)
    # niveau 1.0 = excellent, 0.8 = bien, 0.6 = en progrès
    "Ouala":          ("CABINA",           0.90),
    "Idrissi":        ("CABINA",           0.85),
    "Saadeddine":     ("ENGINE",           0.88),
    "Mana":           ("ENGINE",           0.82),
    "Louz_Mouna":     ("BRIGLIA_UREA",     1.00),  # 100% perf réelle
    "Erradah":        ("COFANO",           0.77),
    "Sour_Bouchra":   ("COFANO",           0.75),
    "Aniba_Nezha":    ("COFANO",           0.67),
    "Fatine":         ("COFANO",           0.67),
    "Soukaina":       ("COFANO",           0.80),
    "Hiba":           ("CONTROLE_ELEC",    0.54),  # 53.8% perf réelle
    "Jaoudallah":     ("CONTROLE_ELEC",    0.80),
    "Mouna_Zair":     ("CONTROLE_ELEC",    0.78),
    "Yadri_Malika":   ("CONTROLE_FINAL",   0.82),
    "Hmidchat":       ("PREMONTAGE",       1.00),  # 100% perf réelle
    "Oumni":          ("PREMONTAGE",       0.85),
    "Alif":           ("PREMONTAGE",       0.88),
    "Babali":         ("PREMONTAGE",       0.87),
    "Hiklef":         ("PREP_GAINE",       0.83),
    "Kenza":          ("EPISSURAGE",       0.80),
    "Badrezzamane":   ("SERTISSAGE",       0.85),
    "Hayat":          ("SERTISSAGE",       0.72),  # 33 retards → impact perf
    "Sougni":         ("SUPPORT_PM",       0.88),
    "Ouizrane":       ("SUPPORT_PM",       0.86),
    "Abdellah":       ("PREP_TABLES",      0.90),
    "Said_Farache":   ("PREP_TABLES",      0.88),
}

# ─────────────────────────────────────────────────────────
# TYPES DE DÉFAUTS réels en faisceau automobile
# (cohérents avec le registre PMSA IVECO)
# ─────────────────────────────────────────────────────────
TYPES_DEFAUT = {
    # Défaut              : (probabilité, zones_à_risque)
    "mauvais_sertissage":   (0.055, ["SERTISSAGE", "COFANO"]),
    "erreur_epissurage":    (0.030, ["EPISSURAGE", "BRIGLIA_UREA"]),
    "longueur_fil_erronee": (0.040, ["ENGINE", "CABINA"]),
    "connecteur_absent":    (0.025, ["CONTROLE_ELEC", "COFANO"]),
    "code_fil_incorrect":   (0.035, ["PREMONTAGE", "ENGINE"]),
    "isolation_endommagee": (0.020, ["PREP_GAINE", "COFANO"]),
    "terminage_incorrect":  (0.045, ["SERTISSAGE", "CABINA"]),
    "plan_non_respecte":    (0.030, ["CONTROLE_FINAL", "PDB"]),
    "conforme":             (0.720, []),  # 72% de conformes
}

PHASES_DETECTION = [
    "sertissage",
    "premontage",
    "controle_electrique",
    "controle_final",
    "non_applicable",  # pour conformes
]


# ─────────────────────────────────────────────────────────
# FONCTION DE GÉNÉRATION D'UN FAISCEAU
# ─────────────────────────────────────────────────────────
def generer_faisceau(harness_id: int) -> dict:
    """
    Génère un faisceau prototype avec données cohérentes SEWS.

    Corrélations métier injectées :
    - Hayat a plus de défauts (33 retards → fatigue)
    - Hiba a moins de conformes (53.8% perf réelle)
    - COFANO plus complexe → plus de défauts (60-85 circuits)
    - ENGINE plus simple → moins de défauts (10-18 circuits)
    """

    # Choix de la famille et de l'opérateur
    famille = random.choice(list(FAMILLES_FAISCEAUX.keys()))
    ref_base, nb_circ_min, nb_circ_max, tps_std = FAMILLES_FAISCEAUX[famille]

    operateur = random.choice(list(OPERATEURS.keys()))
    zone_op, niveau_perf = OPERATEURS[operateur]

    # Date de production (jours ouvrables uniquement)
    delta = (DATE_FIN - DATE_DEBUT).days
    date_prod = DATE_DEBUT + timedelta(days=random.randint(0, delta))
    # Skip weekends
    while date_prod.weekday() >= 5:
        date_prod += timedelta(days=1)

    # Caractéristiques physiques du faisceau
    nb_circuits    = random.randint(nb_circ_min, nb_circ_max)
    nb_connecteurs = int(nb_circuits * random.uniform(0.3, 0.5))
    longueur_m     = round(tps_std * random.uniform(0.8, 1.2), 2)

    # Temps de cycle basé sur le temps standard réel
    # + variation selon l'opérateur et la complexité
    tps_cycle_base = tps_std * 60  # en minutes
    variation      = 1.0 / niveau_perf  # opérateur moins performant → plus long
    tps_cycle      = round(
        np.random.normal(tps_cycle_base * variation, tps_cycle_base * 0.10), 1
    )
    tps_cycle = max(tps_cycle, tps_cycle_base * 0.5)

    # Type de défaut
    # Probabilité ajustée selon le niveau de performance de l'opérateur
    facteur_risque = 1.0 + (1.0 - niveau_perf) * 2
    proba_defauts = {}
    for defaut, (proba, zones_risque) in TYPES_DEFAUT.items():
        if defaut == "conforme":
            continue
        bonus = 1.5 if zone_op in zones_risque else 1.0
        proba_defauts[defaut] = proba * facteur_risque * bonus

    # Normalisation
    total_defaut  = sum(proba_defauts.values())
    proba_conforme = max(0.3, 1.0 - total_defaut)
    proba_conforme = min(proba_conforme, 0.95)
    total_defaut   = 1.0 - proba_conforme

    proba_defauts_norm = {
        k: v / sum(proba_defauts.values()) * total_defaut
        for k, v in proba_defauts.items()
    }
    proba_defauts_norm["conforme"] = proba_conforme

    type_defaut = random.choices(
        list(proba_defauts_norm.keys()),
        weights=list(proba_defauts_norm.values())
    )[0]

    conforme = 1 if type_defaut == "conforme" else 0

    # Phase de détection
    if type_defaut == "conforme":
        phase_detection = "non_applicable"
    elif type_defaut in ["mauvais_sertissage", "terminage_incorrect"]:
        phase_detection = "sertissage"
    elif type_defaut in ["connecteur_absent", "code_fil_incorrect"]:
        phase_detection = "controle_electrique"
    elif type_defaut == "plan_non_respecte":
        phase_detection = "controle_final"
    else:
        phase_detection = random.choice(["premontage", "controle_electrique"])

    # Mesures physiques (cohérentes avec les défauts)
    if conforme:
        hauteur_sertissage = round(np.random.normal(1.85, 0.04), 3)
        force_N            = round(np.random.normal(85, 4), 1)
        resistance_ohm     = round(np.random.normal(0.012, 0.002), 5)
    elif type_defaut in ["mauvais_sertissage", "terminage_incorrect"]:
        ecart = random.choice([-1, 1]) * random.uniform(0.15, 0.40)
        hauteur_sertissage = round(1.85 + ecart, 3)
        force_N            = round(np.random.normal(50, 10), 1)
        resistance_ohm     = round(np.random.normal(0.035, 0.008), 5)
    else:
        hauteur_sertissage = round(np.random.normal(1.85, 0.08), 3)
        force_N            = round(np.random.normal(78, 8), 1)
        resistance_ohm     = round(np.random.normal(0.018, 0.005), 5)

    force_N        = max(force_N, 10.0)
    resistance_ohm = max(resistance_ohm, 0.001)

    # Tests électriques
    if type_defaut == "connecteur_absent":
        test_continuite = 0
        test_cc         = 1
    elif conforme:
        test_continuite = 1
        test_cc         = 1
    else:
        test_continuite = random.choices([1, 0], weights=[0.85, 0.15])[0]
        test_cc         = random.choices([1, 0], weights=[0.90, 0.10])[0]

    # Rebut et retravail
    if conforme:
        rebut = retravail = 0
        tps_retravail = 0.0
    else:
        prob_rebut = {
            "mauvais_sertissage":   0.30,
            "erreur_epissurage":    0.25,
            "longueur_fil_erronee": 0.20,
            "connecteur_absent":    0.10,
            "code_fil_incorrect":   0.15,
            "isolation_endommagee": 0.35,
            "terminage_incorrect":  0.25,
            "plan_non_respecte":    0.40,
        }.get(type_defaut, 0.20)

        rebut     = random.choices([1, 0], weights=[prob_rebut, 1 - prob_rebut])[0]
        retravail = 0 if rebut else 1
        tps_retravail = round(random.uniform(10, 90), 1) if retravail else 0.0

    return {
        "harness_id":                   harness_id,
        "famille":                      famille,
        "reference":                    ref_base,
        "date_production":              date_prod.strftime("%Y-%m-%d"),
        "operateur_id":                 operateur,
        "zone_affectation":             zone_op,
        "nb_circuits":                  nb_circuits,
        "nb_connecteurs":               nb_connecteurs,
        "longueur_totale_m":            longueur_m,
        "temps_standard_h":             tps_std,
        "temps_cycle_min":              tps_cycle,
        "shift":                        "journee",
        "statut_conformite":            conforme,
        "type_defaut":                  type_defaut,
        "phase_detection":              phase_detection,
        "resultat_test_continuite":     test_continuite,
        "resultat_test_court_circuit":  test_cc,
        "hauteur_sertissage_mm":        hauteur_sertissage,
        "force_arrachement_N":          force_N,
        "resistance_ohm":               resistance_ohm,
        "rebut":                        rebut,
        "retravail":                    retravail,
        "temps_retravail_min":          tps_retravail,
    }


# ─────────────────────────────────────────────────────────
# GÉNÉRATION ET SAUVEGARDE
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Partie 1 — Données synthétiques SEWS Cabind (v2)")
    print("=" * 58)
    print(f"  Faisceaux : {NB_FAISCEAUX}")
    print(f"  Période   : {DATE_DEBUT.date()} → {DATE_FIN.date()}")
    print(f"  Familles  : {', '.join(FAMILLES_FAISCEAUX.keys())}")
    print(f"  Opérateurs: {len(OPERATEURS)} (vrais prénoms équipe SEWS)")
    print()

    print("  Génération en cours...")
    lignes = [generer_faisceau(i + 1) for i in range(NB_FAISCEAUX)]
    df = pd.DataFrame(lignes)
    df["date_production"] = pd.to_datetime(df["date_production"])
    df = df.sort_values("date_production").reset_index(drop=True)
    df["harness_id"] = range(1, len(df) + 1)

    # Sauvegarde
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chemin = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(chemin, index=False)

    # ── Statistiques ──
    total      = len(df)
    conformes  = df["statut_conformite"].sum()
    fpy        = conformes / total * 100
    ppm        = (total - conformes) / total * 1_000_000

    print(f"\n  ── Résultats globaux ──")
    print(f"  Total faisceaux : {total:,}")
    print(f"  FPY             : {fpy:.1f}%  (cohérent avec proto SEWS ~75-85%)")
    print(f"  PPM             : {ppm:,.0f}")

    print(f"\n  ── Performance par famille ──")
    par_famille = df.groupby("famille").agg(
        total    =("harness_id", "count"),
        fpy      =("statut_conformite", "mean"),
        tps_moy  =("temps_cycle_min", "mean"),
    ).round(3)
    par_famille["fpy"] = (par_famille["fpy"] * 100).round(1)
    print(par_famille.to_string())

    print(f"\n  ── Top défauts (Pareto) ──")
    defauts = df[df["statut_conformite"] == 0]["type_defaut"].value_counts()
    for d, n in defauts.items():
        print(f"  {d:<28} : {n:>4}  ({n/total*100:.2f}%)")

    print(f"\n  ── Performance par opérateur ──")
    par_op = df.groupby("operateur_id").agg(
        total   =("harness_id", "count"),
        fpy     =("statut_conformite", "mean"),
    ).sort_values("fpy", ascending=False)
    par_op["fpy"] = (par_op["fpy"] * 100).round(1)
    print(par_op.to_string())

    print(f"\n  ✓ Fichier sauvegardé : {chemin}")
    print("=" * 58)
    return df


if __name__ == "__main__":
    df = main()