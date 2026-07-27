"""
data_loader.py — Chargement intelligent des données
Fonctionne dans les 2 modes :
  - LOCAL    : lit depuis sews_reel.db (données réelles)
  - CLOUD    : utilise des données de démonstration intégrées
               (quand la DB n'est pas disponible sur Streamlit Cloud)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

DB_PATH = os.path.join("data", "sews_reel.db")


def mode_cloud():
    """Retourne True si on est sur Streamlit Cloud sans base de données."""
    return not os.path.exists(DB_PATH)


def charger_donnees_demo():
    """
    Données de démonstration intégrées pour Streamlit Cloud.
    Basées sur les vraies données SEWS Cabind (anonymisées).
    """
    np.random.seed(42)

    # ── Opérateurs (vrais prénoms anonymisés) ────────────
    employes_data = {
        "nom_prenom": [
            "Ouala Z.","Idrissi B.","Saadeddine A.","Mana S.",
            "Louz M.","Zaroug F.","Elatmani N.","Erradah",
            "Gheffoub H.","Sour B.","Zaroual F.","Aniba N.",
            "Akourmach A.","Fatine B.","Soukaina I.","Hiba",
            "Jaoudallah N.","Mouna Z.","Yadri M.","Hmidchat A.",
            "Oumni S.","Alif L.","Babali K.","Hiklef S.",
            "Kenza E.","Badrezzamane H.","Hayat","Sougni F.",
            "Ouizrane S.","Abdellah K.","Said F.",
            "Majdoub N.","Marwa E."
        ],
        "matricule": [f"M{str(i).zfill(3)}" for i in range(1, 34)],
        "id_employe": list(range(1, 34)),
    }

    retards_data = {
        "nom_prenom": [
            "Hayat","Hiba","Yadri M.","Sour B.",
            "Zaroual F.","Fatine B.","Elatmani N.",
            "Mana S.","Kenza E.","Oumni S.",
            "Hiklef S.","Zaroug F."
        ],
        "nb_jours_retard": [33, 10, 8, 4, 4, 2, 2, 1, 1, 1, 1, 1],
    }

    affectations_data = {
        "nom_prenom": [
            "Ouala Z.","Idrissi B.","Saadeddine A.","Mana S.",
            "Louz M.","Zaroug F.","Elatmani N.","Erradah",
            "Gheffoub H.","Sour B.","Zaroual F.","Aniba N.",
            "Akourmach A.","Fatine B.","Soukaina I.","Hiba",
            "Jaoudallah N.","Mouna Z.","Yadri M.","Hmidchat A.",
            "Oumni S.","Alif L.","Babali K.","Hiklef S.",
            "Kenza E.","Badrezzamane H.","Hayat","Sougni F.",
            "Ouizrane S.","Abdellah K.","Said F.",
            "Majdoub N.","Marwa E."
        ],
        "zone": [
            "CABINA","CABINA","Engine","Engine",
            "BRIGLIA UREA","COFANO","COFANO","COFANO",
            "COFANO","COFANO","COFANO","COFANO",
            "COFANO","COFANO","COFANO NDE",
            "CONTRÔLE ELECTRIQUE+PIN TO PIN",
            "CONTRÔLE ELECTRIQUE+PIN TO PIN",
            "CONTRÔLE ELECTRIQUE+PIN TO PIN",
            "CONTRÔLE FINAL","PREMONTAGE",
            "PREMONTAGE","PREMONTAGE","PREMONTAGE",
            "PREPARATION GAINE","épissurage",
            "SERTISSAGE +épissurage","SERTISSAGE +épissurage",
            "support PM","support PM",
            "PREPARATION ET VALIDATION LES TABLES DE MONTAGE",
            "PREPARATION ET VALIDATION LES TABLES DE MONTAGE",
            "CABINA","Engine"
        ],
        "tps_std_h": [
            7.5,None,4.0,None,
            2.0,15.0,None,None,
            None,None,None,None,
            None,None,None,
            None,None,None,None,None,
            None,None,None,None,None,
            None,None,None,None,None,
            None,None,None
        ],
        "reference_spn": [
            "P5803561063","P5803561063","P5803686850","P5803686850",
            "P5803638879","P5803549592","P5803549592","P5803549456",
            "P5803549456","P5803549427","P5803549427","P5803563982",
            "P5803563982","P5803657384","P5803621787",
            "ALL","ALL","ALL","ALL","ALL",
            "ALL","ALL","ALL","ALL","ALL",
            "ALL","ALL","ALL","ALL","ALL",
            "ALL","P5803561063","P5803686850"
        ],
        "qty_objectif": [26]*33,
        "qty_reelle":   [26]*33,
        "performance_pct": [100.0]*33,
    }

    # ── Temps standards ──────────────────────────────────
    temps_standards_data = {
        "famille":         ["COFANO","CABINA","PDB","TELAIO H","TELAIO U",
                            "Plafoniera","APH","Engine","BRIGLIA UREA"],
        "tps_proto_h":     [5.76, 2.34, 7.54, 4.84, 3.80, 1.35, 10.23, 1.06, 1.43],
        "tps_coupe_h":     [1.50, 0.80, 2.00, 1.20, 1.00, 0.40,  3.00, 0.30, 0.50],
        "tps_sans_coupe_h":[4.26, 1.54, 5.54, 3.64, 2.80, 0.95,  7.23, 0.76, 0.93],
        "cadence_1op":     [0.5,  0.67, 0.5,  0.67, None, None,  0.17,  2.0,  2.0],
        "cadence_txt":     ["1pièce / 2jour","1pièce / 1,5jour",
                            "1pièce / 2jour","1pièce / 1,5jour",
                            "","","1pièce / 6jour",
                            "2pièce / jour","2pièce / jour"],
    }

    # ── Suivi coupe WK24 ─────────────────────────────────
    suivi_coupe_data = {
        "famille":          ["Cabina Daily My2027","Briglia UREA My2027",
                             "COFANO DAILY MY2027","COFANO DAILY MY2027"],
        "reference":        ["Pr5803561063 00AA","Pr5803607573 00AA",
                             "PR5803621787 02BB","PR5803621806 00A1"],
        "quantite":         [4, 4, 7, 2],
        "nb_reperes":       [148, 67, 487, 486],
        "coupe":            [122, 45, 218, 0],
        "reste":            [26, 22, 269, 486],
        "pct_coupe":        [82.43, 67.16, 44.76, 0.0],
        "date_demande":     ["11/06/2026"]*4,
        "date_reponse_prev":["13/07/2026","13/07/2026","20/07/2026","20/07/2026"],
        "indice_lancement": ["caprow27","brprow27","CF26PROTO","?????"],
    }

    # ── Manhours 2025 ────────────────────────────────────
    mois_list = ["Janvier","Fevrier","Mars","Avril","Mai","Juin",
                 "Juillet","Aout","Septembre","Octobre"]
    projets_mh = [
        ("DAILY","Cofano My",        15.0, 5.8),
        ("DAILY","Briglia  Urea",     2.0, 1.5),
        ("DAILY","Cabina My",         7.5, 2.3),
        ("STRALIS","Engine",          4.0, 1.2),
        ("STRALIS","MOTORE STRALIS",  4.0, 1.2),
        ("DAILY","Dashboard",        18.0, 8.0),
        ("DAILY","TELAIO DAILY2014",  5.0, 3.0),
        ("DAILY","MEAN REAR",         6.0, 3.5),
        ("STRALIS","PARAURTI STRALIS",3.0, 2.0),
        ("DAILY","Telaio Stralis",    5.0, 3.5),
    ]
    mh_rows = []
    np.random.seed(42)
    for projet, sous_proj, tps_proto, tps_serie in projets_mh:
        for m_idx, mois in enumerate(mois_list[:8]):
            qte = np.random.randint(5, 30)
            mh  = round(qte * tps_proto * np.random.uniform(0.85, 1.15), 1)
            mh_rows.append({
                "projet":sous_proj,"sous_projet":sous_proj,
                "mois":mois,"mois_num":m_idx+1,"annee":2025,
                "quantite":float(qte),"manhours":mh,
                "tps_proto_h":tps_proto,"tps_serie_h":tps_serie,
            })

    # ── Anomalies PMSA ───────────────────────────────────
    anomalies_data = {
        "numero":       list(range(1, 9)),
        "statut":       ["Request"]*8,
        "drawing":      [f"5803203185_00_23AZ01{i}" for i in range(1,9)],
        "index_client": ["IVECO"]*8,
        "question":     [
            "Position du connecteur C001 incorrecte selon plan",
            "Longueur faisceau zone moteur trop courte de 15mm",
            "Code couleur fil incorrect — zone tableau de bord",
            "Sertissage terminal T45 non conforme force arrachement",
            "Référence gaine protection moteur erronée",
            "Clip fixation manquant zone châssis gauche",
            "Marquage faisceau illisible après passage four",
            "Connecteur C088 absent du plan assemblage",
        ],
        "solution":     ["En attente réponse IVECO"]*8,
        "jours_attente":[30, 28, 25, 22, 18, 15, 10, 7],
    }

    return {
        "employes":       pd.DataFrame(employes_data),
        "retards":        pd.DataFrame(retards_data),
        "affectations":   pd.DataFrame(affectations_data),
        "temps_standards":pd.DataFrame(temps_standards_data),
        "suivi_coupe":    pd.DataFrame(suivi_coupe_data),
        "ordres_komax":   pd.DataFrame({
            "Description":["COFANO"]*477+["CABINA"]*148+
                          ["BRIGLIA UREA"]*67+["COFANO GS"]*10,
            "est_termine": [1]*385+[0]*317,
            "est_locked":  [1]*251+[0]*451,
            "GoodParts":   np.random.randint(0,50,702),
        }),
        "manhours_2025":  pd.DataFrame(mh_rows),
        "anomalies_pmsa": pd.DataFrame(anomalies_data),
    }


def charger_tables():
    """
    Charge les données depuis SQLite si disponible,
    sinon utilise les données de démonstration.
    Retourne (tables_dict, est_demo: bool)
    """
    if mode_cloud():
        return charger_donnees_demo(), True

    from sqlalchemy import create_engine, text
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    tables = {}
    for nom in ["employes","retards","affectations","temps_standards",
                "suivi_coupe","ordres_komax","manhours_2025","anomalies_pmsa"]:
        try:
            tables[nom] = pd.read_sql(text(f"SELECT * FROM [{nom}]"), engine)
        except:
            tables[nom] = pd.DataFrame()
    return tables, False
