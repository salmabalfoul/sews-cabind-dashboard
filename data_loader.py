"""
data_loader.py — Charge les données pour Streamlit Cloud
Fonctionne avec ou sans base SQLite (fallback sur données démo)

Mode LOCAL  : lit depuis sews_reel.db (données réelles)
Mode CLOUD  : utilise les données de démonstration intégrées
"""

import pandas as pd
import sqlite3
import os

DB_PATH = os.path.join("data", "sews_reel.db")

# ============================================================
# DONNÉES DE DÉMONSTRATION (pour Streamlit Cloud)
# Basées sur les vraies données SEWS Cabind (anonymisées)
# ============================================================

def charger_donnees_demo():
    """
    Retourne des données de démonstration réalistes basées sur SEWS Cabind.
    Ces données sont utilisées quand la base SQLite n'est pas disponible.
    """
    
    # ── 33 EMPLOYÉS AVEC LEURS ZONES ──
    employes = pd.DataFrame({
        "nom_prenom": [
            "Ouala Z.", "Idrissi Bouchra", "Saadine Aziza", "Mana S.",
            "Fatine Bouabdely", "Amina Akourach", "Hasna Gheffoub",
            "Babali K.", "Hiba", "Najat Jaoudallah", "Mouna Zair",
            "Malika Yadri", "Kenza Elwardi", "Aicha Hmidchat",
            "Saadia Oumni", "Alif L.", "Kabbab Abdellah",
            "Said Farache", "Sanaa Hiklef", "Hanane Badrezzamane",
            "Hayat", "Fatimzahra Sougni", "Bouchra Sour",
            "Samira Ouizrane", "Mouna Louz", "Erradah R.",
            "Fatiha Zaroual", "Soukaina Ihikin", "Fatiha Zaroug",
            "Nadia Elatmani", "Oumnia Saadia", "Zair Mouna", "Majdoub N."
        ],
        "matricule": [
            4788, 4647, 5596, 5864, 25121, 3681, 4578, 5444, 4232,
            4745, 4715, 4321, 25122, 3978, 4068, 5066, 14155, 91309,
            4739, 6065, 8795, 5763, 4316, 3809, 4354, 4762, 4671,
            25127, 4261, 4447, 4069, 4710, 5766
        ],
        "zone": [
            "CABINA", "CABINA", "Engine", "Engine", 
            "COFANO", "COFANO", "COFANO", "COFANO", 
            "CONTRÔLE ELECTRIQUE", "CONTRÔLE ELECTRIQUE", "CONTRÔLE ELECTRIQUE",
            "CONTRÔLE FINAL", "épissurage", "PREMONTAGE",
            "PREMONTAGE", "PREMONTAGE", "PREMONTAGE", 
            "PREPARATION ET VALIDATION", "PREPARATION GAINE", 
            "SERTISSAGE", "SERTISSAGE", "support PM",
            "support PM", "COFANO", "BRIGLIA UREA", "COFANO", "COFANO",
            "COFANO NDE", "COFANO", "COFANO", "PREMONTAGE", "CONTRÔLE ELECTRIQUE", "COFANO"
        ]
    })
    
    # ── RETARDS PAR EMPLOYÉ ──
    retards = pd.DataFrame({
        "nom_prenom": [
            "Ouala Z.", "Idrissi Bouchra", "Saadine Aziza", "Mana S.",
            "Fatine Bouabdely", "Amina Akourach", "Hasna Gheffoub",
            "Babali K.", "Hiba", "Najat Jaoudallah", "Mouna Zair",
            "Malika Yadri", "Kenza Elwardi", "Aicha Hmidchat",
            "Saadia Oumni", "Alif L.", "Kabbab Abdellah",
            "Said Farache", "Sanaa Hiklef", "Hanane Badrezzamane",
            "Hayat", "Fatimzahra Sougni", "Bouchra Sour",
            "Samira Ouizrane", "Mouna Louz", "Erradah R.",
            "Fatiha Zaroual", "Soukaina Ihikin", "Fatiha Zaroug",
            "Nadia Elatmani", "Oumnia Saadia", "Zair Mouna", "Majdoub N."
        ],
        "nb_jours_retard": [
            0, 0, 0, 0, 2, 0, 0, 0, 10, 0, 0, 8, 1, 0, 1, 0,
            0, 0, 1, 0, 33, 0, 4, 0, 1, 0, 4, 0, 1, 2, 0, 0, 0
        ]
    })
    
    # ── AFFECTATIONS DES OPÉRATEURS ──
    affectations = pd.DataFrame({
        "nom_prenom": [
            "Ouala Z.", "Idrissi Bouchra", "Saadine Aziza", "Mana S.",
            "Louz Mouna", "Fatiha Zaroug", "Nadia Elatmani",
            "Erradah R.", "Hasna Gheffoub", "Bouchra Sour",
            "Fatiha Zaroual", "Amina Akourach", "Fatine Bouabdely",
            "Soukaina Ihikin", "Hiba", "Najat Jaoudallah",
            "Mouna Zair", "Malika Yadri", "Aicha Hmidchat",
            "Saadia Oumni", "Alif L.", "Babali K.",
            "Sanaa Hiklef", "Kenza Elwardi", "Hanane Badrezzamane",
            "Hayat", "Fatimzahra Sougni", "Samira Ouizrane",
            "Kabbab Abdellah", "Said Farache"
        ],
        "zone": [
            "CABINA", "CABINA", "Engine", "Engine", 
            "BRIGLIA UREA", "COFANO", "COFANO", "COFANO", "COFANO", 
            "COFANO", "COFANO", "COFANO", "COFANO", "COFANO NDE",
            "CONTRÔLE ELECTRIQUE", "CONTRÔLE ELECTRIQUE", "CONTRÔLE ELECTRIQUE",
            "CONTRÔLE FINAL", "PREMONTAGE", "PREMONTAGE", "PREMONTAGE", "PREMONTAGE",
            "PREPARATION GAINE", "épissurage", "SERTISSAGE",
            "SERTISSAGE", "support PM", "support PM",
            "PREPARATION ET VALIDATION", "PREPARATION ET VALIDATION"
        ],
        "reference_spn": [
            "P5803561063/88 00AA", "P5803561063/88 00AA",
            "P5803686850 00AA", "P5803686850 00AA",
            "P5803638879/82 00AA", "P5803549592 01AA",
            "P5803549592 01AA", "P5803549456 02AA",
            "P5803549456 02AA", "P5803549427 02AA",
            "P5803549427 02AA", "P5803563982/56 01A1",
            "P5803657384 00AA", "P5803621787 00A0", "ALL",
            "ALL", "ALL", "ALL", "ALL", "ALL", "ALL", "ALL",
            "ALL", "ALL", "ALL", "ALL", "ALL", "ALL", "ALL", "ALL"
        ]
    })
    
    # ── TEMPS STANDARDS PAR FAMILLE ──
    temps_standards = pd.DataFrame({
        "famille": ["COFANO", "CABINA", "ENGINE", "BRIGLIA UREA", "PDB",
                   "TELAIO H", "TELAIO U", "Plafoniera", "APH"],
        "tps_proto_h": [5.76, 2.34, 1.06, 1.43, 7.54, 4.84, 3.80, 1.35, 10.23],
        "cadence_txt": [
            "1pièce / 2jour", "1pièce / 1,5jour", "2pièce / jour",
            "2pièce / jour", "1pièce / 2jour", "1pièce / 1,5jour",
            "", "", "1pièce / 6jour"
        ]
    })
    
    # ── SUIVI DE LA COUPE ──
    suivi_coupe = pd.DataFrame({
        "famille": [
            "Cabina Daily My2027", "Briglia UREA My2027",
            "COFANO DAILY MY2027", "COFANO DAILY MY2027"
        ],
        "reference": [
            "Pr5803561063 00AA", "Pr5803607573 00AA",
            "PR5803621787 02BB", "PR5803621806 00A1"
        ],
        "nb_reperes": [148, 67, 487, 486],
        "coupe": [122, 45, 218, 0],
        "reste": [26, 22, 269, 486],
        "pct_coupe": [82.4, 67.2, 44.8, 0.0],
        "date_demande": [
            "2025-06-02", "2025-06-02", "2025-06-02", "2025-06-02"
        ],
        "date_reponse_prev": [
            "2025-06-09", "2025-06-09", "2025-06-09", "2025-06-09"
        ],
        "indice_lancement": ["LNC-001", "LNC-002", "LNC-003", "?????"]
    })
    
    # ── ORDRES KOMAX ──
    ordres_komax = pd.DataFrame({
        "Description": ["COFANO", "CABINA", "BRIGLIA UREA", "COFANO GS"],
        "est_termine": [1, 1, 0, 0],
        "est_locked": [0, 1, 1, 0]
    })
    
    # ── MANHOURS 2025 ──
    manhours_2025 = pd.DataFrame({
        "sous_projet": [
            "MOTORE STRALIS", "Engine", "Cabina My", "COFANO",
            "TELAIO DAILY2014", "Cofano My", "Dashboard",
            "MEAN REAR", "PARAURTI STRALIS", "Briglia Urea",
            "Telaio Stralis"
        ],
        "quantite": [45, 357, 54, 40, 85, 66, 112, 11, 90, 83, 5],
        "manhours": [
            90.0, 428.4, 124.2, 196.8, 130.1, 382.8, 840.0, 112.2, 99.0, 116.2, 24.0
        ],
        "tps_proto_h": [7.0, 1.2, 2.3, 4.9, 1.6, 5.8, 7.4, 10.2, 1.0, 1.4, 4.0]
    })
    
    # ── ANOMALIES PMSA (8 anomalies) ──
    anomalies_pmsa = pd.DataFrame({
        "numero": [1, 2, 3, 4, 5, 6, 7, 8],
        "statut": ["Request"] * 8,
        "drawing": ["5803203185"] * 8,
        "jours_attente": [30, 25, 20, 15, 10, 8, 5, 3],
        "status": ["Request"] * 8
    })
    
    return {
        "employes": employes,
        "retards": retards,
        "affectations": affectations,
        "temps_standards": temps_standards,
        "suivi_coupe": suivi_coupe,
        "ordres_komax": ordres_komax,
        "manhours_2025": manhours_2025,
        "anomalies_pmsa": anomalies_pmsa
    }


# ============================================================
# FONCTION PRINCIPALE DE CHARGEMENT
# ============================================================

def charger_tables():
    """
    Charge les données depuis SQLite si disponible, sinon utilise les données démo.
    
    Retourne :
        tables (dict) : Dictionnaire des DataFrames
        est_demo (bool) : True si données de démonstration, False si données réelles
    """
    
    # Si la base SQLite existe, la charger
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            tables = {}
            
            for nom in ["employes", "retards", "affectations", "temps_standards",
                        "suivi_coupe", "ordres_komax", "manhours_2025", "anomalies_pmsa"]:
                try:
                    tables[nom] = pd.read_sql_query(f"SELECT * FROM {nom}", conn)
                except:
                    tables[nom] = pd.DataFrame()
            
            conn.close()
            
            # Vérifier que les données ne sont pas vides
            if tables.get("employes", pd.DataFrame()).empty:
                return charger_donnees_demo(), True
            
            return tables, False
            
        except Exception:
            return charger_donnees_demo(), True
    
    # Si la base n'existe pas, utiliser les données démo
    return charger_donnees_demo(), True


# ============================================================
# FONCTION D'EXPORT VERS SQLITE (pour usage local)
# ============================================================

def exporter_donnees_demo():
    """
    Exporte les données de démonstration vers SQLite.
    Utile pour créer une base de départ sur le PC local.
    """
    tables, _ = charger_tables()
    
    conn = sqlite3.connect(DB_PATH)
    for nom, df in tables.items():
        df.to_sql(nom, conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"✅ Données de démonstration exportées vers {DB_PATH}")


# ============================================================
# POINT D'ENTRÉE (pour tester)
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  data_loader.py — Test de chargement")
    print("=" * 55)
    
    tables, est_demo = charger_tables()
    
    print(f"\n📊 Mode : {'DÉMONSTRATION' if est_demo else 'RÉEL'}")
    print(f"📁 Tables chargées : {len(tables)}")
    
    for nom, df in tables.items():
        print(f"  ✅ {nom:<20} : {len(df):>5} lignes")
    
    print("\n" + "=" * 55)