"""
ETL Réel — SEWS Cabind (Version 2)
Cherche les fichiers Excel dans 'donnees_reelles/' EN PRIORITÉ
puis dans le dossier courant si non trouvé.
Compatible avec le LANCER_DASHBOARD.bat
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import os, sys, warnings
warnings.filterwarnings("ignore")

DB_PATH = os.path.join("data", "sews_reel.db")
os.makedirs("data", exist_ok=True)
os.makedirs("donnees_reelles", exist_ok=True)

# Noms des fichiers Excel
FICHIERS = {
    "employes":    "Employee_retard_schedule.xlsx",
    "affectation": "Proto_team_affectation_dmx.xlsx",
    "temps_std":   "Temps_standard_PROTO.xlsx",
    "planning":    "Planning_Proto_PB_DAYLI_MY27_.xlsx",
    "anomalies":   "PMSA_5803203185_00_23AZ013.xlsx",
    "coupe":       "situation_de_la_coupe_wk24.xlsx",
    "manhours":    "Classeur3.xlsx",
}

def trouver_fichier(nom_fichier):
    """
    Cherche le fichier dans cet ordre :
    1. donnees_reelles/nom_fichier  ← priorité
    2. nom_fichier (dossier courant)
    Retourne le chemin trouvé ou None.
    """
    chemin1 = os.path.join("donnees_reelles", nom_fichier)
    chemin2 = nom_fichier
    if os.path.exists(chemin1):
        return chemin1
    if os.path.exists(chemin2):
        return chemin2
    return None


def verifier_fichiers():
    print("\n── Vérification des fichiers ───────────────────")
    tous_ok = True
    for cle, nom in FICHIERS.items():
        chemin = trouver_fichier(nom)
        if chemin:
            print(f"  ✓  {nom}  ({chemin})")
        else:
            print(f"  ✗ MANQUANT  {nom}")
            if cle not in ["planning"]:  # planning optionnel
                tous_ok = False
    if not tous_ok:
        print("\n  ⚠ Place les fichiers manquants dans 'donnees_reelles/'")
    return tous_ok


def extraire_employes_retards():
    print("\n── Extraction employés et retards ─────────────")
    fichier = trouver_fichier(FICHIERS["employes"])
    if not fichier:
        return None, None

    df_team = pd.read_excel(fichier, sheet_name="team", header=0)
    df_team.columns = ["nom_prenom","matricule"]
    df_team = df_team.dropna(subset=["matricule"]).copy()
    df_team["matricule"]  = df_team["matricule"].astype(str).str.strip()
    df_team["nom_prenom"] = df_team["nom_prenom"].astype(str).str.strip()
    df_team["id_employe"] = range(1, len(df_team)+1)
    print(f"  ✓ {len(df_team)} opérateurs extraits")

    df_r = pd.read_excel(fichier, sheet_name="Feuil1", header=None)
    retards = df_r.iloc[4:, [3,4]].copy()
    retards.columns = ["nom_prenom","nb_jours_retard"]
    retards = retards.dropna(subset=["nom_prenom"]).copy()
    retards["nom_prenom"] = (retards["nom_prenom"].astype(str)
                             .str.replace('\xa0',' ',regex=False).str.strip())
    retards["nb_jours_retard"] = (pd.to_numeric(retards["nb_jours_retard"],errors="coerce")
                                  .fillna(0).astype(int))
    retards = retards[retards["nom_prenom"] != "Employee name"].reset_index(drop=True)
    print(f"  ✓ {len(retards)} entrées de retard")
    return df_team, retards


def extraire_affectations():
    print("\n── Extraction affectations ─────────────────────")
    fichier = trouver_fichier(FICHIERS["affectation"])
    if not fichier:
        return None

    df_aff = pd.read_excel(fichier, sheet_name="Feuil1", header=None)
    rows = []
    for i in range(6, min(37, df_aff.shape[0])):
        row = df_aff.iloc[i, :8].tolist()
        nom  = str(row[0]).replace('\xa0',' ').strip()
        zone = str(row[1]).replace('\xa0',' ').strip()
        tps  = row[2] if pd.notna(row[2]) else None
        spn  = str(row[3]).strip() if pd.notna(row[3]) else ""
        qty_obj  = row[4] if pd.notna(row[4]) else 0
        qty_real = sum(int(row[j]) for j in [5,6,7] if pd.notna(row[j]))
        if nom and nom not in ["nan",""]:
            rows.append({
                "nom_prenom":   nom,
                "zone":         zone if zone != "nan" else "",
                "tps_std_h":    float(tps) if tps else None,
                "reference_spn":spn,
                "qty_objectif": int(qty_obj) if pd.notna(qty_obj) else 0,
                "qty_reelle":   qty_real,
                "performance_pct": round(qty_real/float(qty_obj)*100,1)
                               if qty_obj and float(qty_obj)>0 and qty_real>0 else 0.0,
            })
    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df)} affectations extraites")
    return df


def extraire_temps_standards():
    print("\n── Extraction temps standards ──────────────────")
    fichier = trouver_fichier(FICHIERS["temps_std"])
    if not fichier:
        return None

    df_ts = pd.read_excel(fichier, sheet_name="Feuil 1", header=None)
    projets = []
    for i in range(7, 16):
        if i >= df_ts.shape[0]:
            break
        row = df_ts.iloc[i, :10].tolist()
        projet = str(row[1]).strip() if pd.notna(row[1]) else ""
        if projet and projet not in ["nan",""]:
            projets.append({
                "famille":          projet,
                "tps_proto_h":      float(row[2]) if pd.notna(row[2]) else None,
                "tps_coupe_h":      float(row[3]) if pd.notna(row[3]) else None,
                "tps_sans_coupe_h": float(row[4]) if pd.notna(row[4]) else None,
                "cadence_1op":      float(row[8]) if pd.notna(row[8]) else None,
                "cadence_txt":      str(row[9]).strip() if pd.notna(row[9]) else "",
            })
    df = pd.DataFrame(projets)
    print(f"  ✓ {len(df)} familles de faisceaux")
    return df


def extraire_coupe():
    print("\n── Extraction suivi de coupe ───────────────────")
    fichier = trouver_fichier(FICHIERS["coupe"])
    if not fichier:
        return None, None

    df_raw = pd.read_excel(fichier, sheet_name="Suivie de la coupe", header=None)
    rows = []
    for i in range(1, df_raw.shape[0]):
        row = df_raw.iloc[i, :10].tolist()
        if pd.notna(row[0]) and str(row[0]).strip() not in ["","nan"]:
            pct_raw = row[6]
            pct = float(pct_raw)*100 if pd.notna(pct_raw) and float(pct_raw)<=1 \
                  else float(pct_raw) if pd.notna(pct_raw) else 0.0
            rows.append({
                "famille":          str(row[0]).strip(),
                "reference":        str(row[1]).strip(),
                "quantite":         int(row[2]) if pd.notna(row[2]) else 0,
                "nb_reperes":       int(row[3]) if pd.notna(row[3]) else 0,
                "coupe":            int(row[4]) if pd.notna(row[4]) else 0,
                "reste":            int(row[5]) if pd.notna(row[5]) else 0,
                "pct_coupe":        round(pct,2),
                "date_demande":     str(row[7])[:10] if pd.notna(row[7]) else "",
                "date_reponse_prev":str(row[8])[:10] if pd.notna(row[8]) else "",
                "indice_lancement": str(row[9]).strip() if pd.notna(row[9]) else "?????",
            })
    df_coupe = pd.DataFrame(rows)
    print(f"  ✓ {len(df_coupe)} faisceaux en suivi de coupe")

    df_kx = pd.read_excel(fichier, sheet_name="file (4)", header=0)
    df_kx.columns = [c.strip() for c in df_kx.columns]
    for col in ["TargetDate","PlannedStartDate","PlannedEndDate","ActStartDate","ActEndDate"]:
        if col in df_kx.columns:
            df_kx[col] = pd.to_datetime(df_kx[col], errors="coerce").astype(str)
    df_kx["est_termine"] = (df_kx["ActEndDate"].notna() &
                            (df_kx["ActEndDate"] != "NaT")).astype(int)
    df_kx["est_locked"]  = (df_kx["locked"] == "yes").astype(int)
    df_kx["GoodParts"]   = pd.to_numeric(df_kx["GoodParts"], errors="coerce").fillna(0)
    print(f"  ✓ {len(df_kx)} ordres KOMAX")
    return df_coupe, df_kx


def extraire_manhours():
    print("\n── Extraction manhours 2025 ────────────────────")
    fichier = trouver_fichier(FICHIERS["manhours"])
    if not fichier:
        return None

    df_mh = pd.read_excel(fichier, sheet_name="pour 2025", header=None)
    mois_noms = ["Janvier","Fevrier","Mars","Avril","Mai","Juin",
                 "Juillet","Aout","Septembre","Octobre"]
    rows = []
    projet_courant = ""
    for i in range(3, 16):
        if i >= df_mh.shape[0]:
            break
        row = df_mh.iloc[i, :].tolist()
        if pd.notna(row[0]) and str(row[0]).strip() not in ["nan","Total per Month",""]:
            projet_courant = str(row[0]).strip()
        sous_projet = str(row[1]).strip() if pd.notna(row[1]) else ""
        tps_proto   = float(row[2]) if pd.notna(row[2]) else None
        if sous_projet and sous_projet not in ["nan","Sub-Project","Total per Month",""]:
            for m_idx, mois in enumerate(mois_noms):
                col_qte = 4+m_idx*2
                col_mh  = 5+m_idx*2
                qte = float(row[col_qte]) if col_qte<len(row) and pd.notna(row[col_qte]) else 0.0
                mh  = float(row[col_mh])  if col_mh <len(row) and pd.notna(row[col_mh])  else 0.0
                if qte>0 or mh>0:
                    rows.append({
                        "projet":projet_courant,"sous_projet":sous_projet,
                        "mois":mois,"mois_num":m_idx+1,"annee":2025,
                        "quantite":qte,"manhours":mh,"tps_proto_h":tps_proto,
                    })
    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df)} enregistrements manhours")
    return df


def extraire_anomalies():
    print("\n── Extraction anomalies PMSA ───────────────────")
    fichier = trouver_fichier(FICHIERS["anomalies"])
    if not fichier:
        return None

    df_pm = pd.read_excel(fichier, sheet_name="PMSA_23AZ013", header=None)
    rows = []
    for i in range(5, df_pm.shape[0]):
        row = df_pm.iloc[i, :24].tolist()
        if pd.notna(row[0]):
            question = solution = ""
            for j in range(8,24):
                val = str(df_pm.iloc[i,j]).strip() if pd.notna(df_pm.iloc[i,j]) else ""
                if len(val)>20 and not question:
                    question = val[:300]
                elif len(val)>10 and question and not solution:
                    solution = val[:300]
            rows.append({
                "numero":       int(row[0]),
                "statut":       str(row[1]).strip() if pd.notna(row[1]) else "Request",
                "drawing":      str(row[2]).strip() if pd.notna(row[2]) else "",
                "index_client": str(row[3]).strip() if pd.notna(row[3]) else "",
                "question":     question,
                "solution":     solution,
                "jours_attente":30,
            })
    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df)} anomalies — Statuts: {df['statut'].value_counts().to_dict()}")
    return df


def charger_sqlite(tables):
    print("\n── Chargement dans SQLite ──────────────────────")
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    for nom, df in tables.items():
        if df is not None and len(df)>0:
            df.to_sql(nom, engine, if_exists="replace", index=False)
            print(f"  ✓ {nom:<25} : {len(df):>5} lignes")
        else:
            print(f"  ⚠ {nom:<25} : vide ou manquant")
    print("\n── Vérification finale ─────────────────────────")
    with engine.connect() as conn:
        tbls = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )).fetchall()
        for (t,) in tbls:
            n = conn.execute(text(f"SELECT COUNT(*) FROM [{t}]")).fetchone()[0]
            print(f"  {t:<25} : {n:>5} lignes ✓")


def main():
    print("="*55)
    print("  ETL Réel — SEWS Cabind")
    print(f"  {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
    print("="*55)

    verifier_fichiers()

    df_emp, df_ret = extraire_employes_retards()
    df_aff         = extraire_affectations()
    df_tps         = extraire_temps_standards()
    df_coupe, df_kx= extraire_coupe()
    df_mh          = extraire_manhours()
    df_anom        = extraire_anomalies()

    charger_sqlite({
        "employes":       df_emp,
        "retards":        df_ret,
        "affectations":   df_aff,
        "temps_standards":df_tps,
        "suivi_coupe":    df_coupe,
        "ordres_komax":   df_kx,
        "manhours_2025":  df_mh,
        "anomalies_pmsa": df_anom,
    })

    print("\n"+"="*55)
    print(f"  ✓ Base créée : {DB_PATH}")
    print("="*55)

if __name__ == "__main__":
    main()