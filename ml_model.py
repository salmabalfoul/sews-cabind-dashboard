"""
Phase 4 — Modèle prédictif de défauts
Projet : Système intelligent de pilotage des faisceaux — SEWS Cabind

Pipeline complet :
  1. Préparation des features
  2. Train / Test split stratifié
  3. Entraînement : Logistic Regression, Random Forest, XGBoost
  4. Évaluation : Accuracy, F1, Matrice de confusion, ROC
  5. Interprétabilité SHAP
  6. Sauvegarde du meilleur modèle (model.pkl)
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import xgboost as xgb
import shap

from db_config import get_url

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
SEED         = 42
TEST_SIZE    = 0.20      # 20% pour le test
OUTPUT_DIR   = "models"  # dossier de sauvegarde
MODEL_FILE   = os.path.join(OUTPUT_DIR, "model.pkl")
ENCODER_FILE = os.path.join(OUTPUT_DIR, "encoders.pkl")

# Les 9 features sélectionnées (mesurées AVANT le résultat final)
FEATURES = [
    "hauteur_sertissage_mm",   # mesure physique critique
    "force_arrachement_N",     # qualité de la connexion
    "resistance_ohm",          # qualité électrique
    "temps_cycle_min",         # durée de fabrication
    "nb_circuits",             # complexité du faisceau
    "nb_connecteurs",          # complexité du faisceau
    "longueur_totale_m",       # taille du faisceau
    "shift_encoded",           # shift encodé numériquement
    "operateur_encoded",       # opérateur encodé numériquement
]

TARGET = "statut_conformite"   # 1 = conforme, 0 = défaut


# ═══════════════════════════════════════════
# ÉTAPE 1 — CHARGEMENT ET PRÉPARATION DES FEATURES
# ═══════════════════════════════════════════
def preparer_features() -> tuple:
    """
    Charge les données depuis SQLite et prépare les features ML.

    Encodage des variables catégorielles :
    - shift : matin=0, apres_midi=1, nuit=2
    - operateur_id : OP_001=0, OP_002=1, ..., OP_020=19

    Retourne : X (features), y (cible), encoders (pour réutilisation)
    """
    print("\n── ÉTAPE 1 : Préparation des features ──────")

    # Chargement depuis SQLite
    engine = create_engine(get_url(), echo=False)
    df = pd.read_sql(
        text("SELECT * FROM production_harness"),
        engine
    )
    print(f"  Données chargées : {len(df):,} faisceaux")

    # Encodage de 'shift' (variable ordinale : matin < apres_midi < nuit)
    shift_map = {"matin": 0, "apres_midi": 1, "nuit": 2}
    df["shift_encoded"] = df["shift"].map(shift_map).fillna(0).astype(int)

    # Encodage de 'operateur_id' (variable nominale)
    le_operateur = LabelEncoder()
    df["operateur_encoded"] = le_operateur.fit_transform(df["operateur_id"])

    # Sauvegarde des encodeurs pour les réutiliser en prédiction
    encoders = {
        "shift_map":    shift_map,
        "le_operateur": le_operateur,
    }

    # Sélection des features et de la cible
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Vérification des valeurs manquantes
    nb_na = X.isnull().sum().sum()
    if nb_na > 0:
        print(f"  ⚠ {nb_na} valeurs manquantes → remplacement par médiane")
        X = X.fillna(X.median())

    print(f"  Features sélectionnées : {len(FEATURES)}")
    print(f"  Distribution de la cible :")
    print(f"    Conformes (1)     : {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"    Non-conformes (0) : {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")
    print(f"  ✓ Features prêtes")

    return X, y, encoders


# ═══════════════════════════════════════════
# ÉTAPE 2 — DÉCOUPAGE TRAIN / TEST
# ═══════════════════════════════════════════
def decouper_donnees(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Découpe en 80% entraînement / 20% test.
    Stratifié : même proportion de défauts dans les deux parties.
    """
    print("\n── ÉTAPE 2 : Découpage train / test ────────")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y          # garantit la même proportion de défauts
    )

    print(f"  Entraînement : {len(X_train):,} faisceaux  "
          f"({y_train.mean()*100:.1f}% conformes)")
    print(f"  Test         : {len(X_test):,} faisceaux  "
          f"({y_test.mean()*100:.1f}% conformes)")
    print(f"  ✓ Découpage stratifié effectué")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════
# ÉTAPE 3 — ENTRAÎNEMENT DES MODÈLES
# ═══════════════════════════════════════════
def entrainer_modeles(X_train, y_train) -> dict:
    """
    Entraîne 3 modèles dans l'ordre croissant de complexité.

    Régression logistique → modèle baseline simple et interprétable
    Random Forest         → robuste, gère les non-linéarités
    XGBoost               → généralement le plus performant

    Retourne un dictionnaire {nom: modèle entraîné}
    """
    print("\n── ÉTAPE 3 : Entraînement des modèles ──────")

    # Normalisation pour la régression logistique
    # (XGBoost et RF n'en ont pas besoin, mais ça ne nuit pas)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modeles = {}

    # ── Modèle 1 : Régression Logistique (baseline) ──
    print("\n  [1/3] Régression Logistique...")
    rl = LogisticRegression(
        random_state=SEED,
        max_iter=1000,
        class_weight="balanced"  # compense le déséquilibre 73/27
    )
    rl.fit(X_train_scaled, y_train)
    modeles["Logistic Regression"] = {
        "model":  rl,
        "scaler": scaler,
        "scaled": True,
    }
    print(f"  ✓ Régression logistique entraînée")

    # ── Modèle 2 : Random Forest ──
    print("\n  [2/3] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,          # 200 arbres de décision
        max_depth=10,              # profondeur max pour éviter le surapprentissage
        min_samples_leaf=5,        # au moins 5 exemples par feuille
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1                  # utilise tous les cœurs CPU
    )
    rf.fit(X_train, y_train)
    modeles["Random Forest"] = {
        "model":  rf,
        "scaler": None,
        "scaled": False,
    }
    print(f"  ✓ Random Forest entraîné (200 arbres)")

    # ── Modèle 3 : XGBoost ──
    print("\n  [3/3] XGBoost...")
    # Calcul du ratio pour compenser le déséquilibre de classes
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,    # compense 73% conformes vs 27% défauts
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    modeles["XGBoost"] = {
        "model":  xgb_model,
        "scaler": None,
        "scaled": False,
    }
    print(f"  ✓ XGBoost entraîné (300 estimateurs)")

    return modeles


# ═══════════════════════════════════════════
# ÉTAPE 4 — ÉVALUATION DES PERFORMANCES
# ═══════════════════════════════════════════
def evaluer_modeles(modeles: dict, X_test, y_test) -> dict:
    """
    Évalue chaque modèle sur les données de test (jamais vues).

    Métriques calculées :
    - Accuracy  : % de prédictions correctes (toutes classes)
    - F1-score  : équilibre précision/rappel (important pour les défauts)
    - AUC-ROC   : capacité à discriminer conforme vs défaut
    - Matrice de confusion : détail des erreurs par type
    """
    print("\n── ÉTAPE 4 : Évaluation des performances ───")
    print(f"\n  {'Modèle':<22} {'Accuracy':>9} {'F1':>8} {'AUC-ROC':>9}")
    print(f"  {'-'*52}")

    resultats = {}

    for nom, info in modeles.items():
        model  = info["model"]
        scaled = info["scaled"]

        # Prédiction
        X_eval = info["scaler"].transform(X_test) if scaled else X_test
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        # Métriques
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        auc    = roc_auc_score(y_test, y_prob)
        cm     = confusion_matrix(y_test, y_pred)

        resultats[nom] = {
            "accuracy": round(acc, 4),
            "f1":       round(f1, 4),
            "auc":      round(auc, 4),
            "cm":       cm,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
        }

        print(f"  {nom:<22} {acc*100:>8.2f}%  {f1:>7.4f}  {auc:>8.4f}")

    # Détail du meilleur modèle
    meilleur = max(resultats, key=lambda k: resultats[k]["f1"])
    print(f"\n  Meilleur modèle : {meilleur}  (F1 = {resultats[meilleur]['f1']:.4f})")

    # Matrice de confusion du meilleur
    cm = resultats[meilleur]["cm"]
    print(f"\n  Matrice de confusion — {meilleur} :")
    print(f"                    Prédit défaut  Prédit conforme")
    print(f"  Réel défaut     :      {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Réel conforme   :      {cm[1][0]:>5}          {cm[1][1]:>5}")

    vrai_positif  = cm[0][0]
    faux_negatif  = cm[0][1]
    faux_positif  = cm[1][0]
    print(f"\n  Interprétation :")
    print(f"  → {vrai_positif} défauts correctement détectés")
    print(f"  → {faux_negatif} défauts manqués (faux négatifs — à minimiser !)")
    print(f"  → {faux_positif} conformes signalés à tort (faux positifs — acceptable)")

    # Rapport complet scikit-learn
    print(f"\n  Rapport de classification — {meilleur} :")
    print(classification_report(
        y_test,
        resultats[meilleur]["y_pred"],
        target_names=["Défaut (0)", "Conforme (1)"]
    ))

    return resultats, meilleur


# ═══════════════════════════════════════════
# ÉTAPE 5 — INTERPRÉTABILITÉ SHAP
# ═══════════════════════════════════════════
def analyser_shap(modele_info: dict, X_train, X_test) -> None:
    """
    Analyse SHAP : explique l'importance des features.

    SHAP (SHapley Additive exPlanations) calcule la contribution
    de chaque feature à chaque prédiction individuelle.

    Affiche :
    - L'importance globale de chaque feature
    - L'explication d'une prédiction individuelle (exemple)
    """
    print("\n── ÉTAPE 5 : Analyse SHAP ──────────────────")

    model = modele_info["model"]

    # Explainer SHAP pour XGBoost (TreeExplainer est le plus rapide)
    print("  Calcul des valeurs SHAP...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Importance globale : moyenne des valeurs SHAP absolues
    importance_globale = pd.DataFrame({
        "feature":    FEATURES,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)

    print("\n  Importance globale des features (SHAP) :")
    print(f"  {'Feature':<28} {'Importance SHAP':>16}")
    print(f"  {'-'*46}")
    for _, row in importance_globale.iterrows():
        barre = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<28} {row['importance']:>10.4f}  {barre}")

    # Explication d'une prédiction individuelle
    # Prend le premier faisceau défectueux trouvé dans le test
    X_test_df = pd.DataFrame(X_test, columns=FEATURES)
    y_test_array = np.array(modele_info.get("y_test_ref", [0]))

    print("\n  Exemple d'explication individuelle :")
    print("  (faisceau n°0 de l'ensemble de test)")
    exemple_shap = shap_values[0]
    exemple_feat = X_test_df.iloc[0]

    contributions = pd.DataFrame({
        "feature":      FEATURES,
        "valeur":       exemple_feat.values,
        "contribution": exemple_shap
    }).sort_values("contribution", key=abs, ascending=False)

    for _, row in contributions.iterrows():
        signe  = "+" if row["contribution"] > 0 else "-"
        impact = "→ vers DEFAUT" if row["contribution"] < 0 else "→ vers CONFORME"
        print(f"    {row['feature']:<28} = {row['valeur']:>8.3f}  "
              f"SHAP: {signe}{abs(row['contribution']):.4f}  {impact}")

    print("\n  ✓ Analyse SHAP terminée")
    return importance_globale


# ═══════════════════════════════════════════
# SAUVEGARDE DU MEILLEUR MODÈLE
# ═══════════════════════════════════════════
def sauvegarder_modele(modeles: dict, meilleur: str, encoders: dict) -> None:
    """
    Sauvegarde le meilleur modèle en fichier .pkl.
    Ce fichier sera utilisé directement par le dashboard (Phase 5).

    Contenu du fichier sauvegardé :
    - Le modèle entraîné
    - Le scaler (si utilisé)
    - Les encodeurs (shift_map, le_operateur)
    - La liste des features dans le bon ordre
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paquet = {
        "modele":   modeles[meilleur]["model"],
        "scaler":   modeles[meilleur]["scaler"],
        "scaled":   modeles[meilleur]["scaled"],
        "features": FEATURES,
        "encoders": encoders,
        "nom":      meilleur,
        "version":  "1.0",
        "date":     pd.Timestamp.now().strftime("%Y-%m-%d"),
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(paquet, f)

    taille = os.path.getsize(MODEL_FILE) / 1024
    print(f"\n  Modèle sauvegardé : {MODEL_FILE}  ({taille:.0f} Ko)")
    print(f"  Contenu : modèle {meilleur} + encodeurs + liste des features")


# ═══════════════════════════════════════════
# FONCTION DE PRÉDICTION — pour le dashboard
# ═══════════════════════════════════════════
def predire(
    hauteur_sertissage_mm: float,
    force_arrachement_N:   float,
    resistance_ohm:        float,
    temps_cycle_min:       float,
    nb_circuits:           int,
    nb_connecteurs:        int,
    longueur_totale_m:     float,
    shift:                 str,
    operateur_id:          str,
    chemin_modele:         str = MODEL_FILE,
) -> dict:
    """
    Prédit si un nouveau faisceau sera défectueux.
    C'est cette fonction qui sera appelée par le dashboard Streamlit.

    Retourne :
    - prediction   : 1 (conforme) ou 0 (défaut prédit)
    - probabilite  : probabilité de conformité (0 à 1)
    - risque       : 'faible' / 'modéré' / 'élevé'
    - recommandation : action suggérée
    """
    # Chargement du modèle
    with open(chemin_modele, "rb") as f:
        paquet = pickle.load(f)

    model    = paquet["modele"]
    scaler   = paquet["scaler"]
    scaled   = paquet["scaled"]
    encoders = paquet["encoders"]

    # Encodage du shift
    shift_encoded = encoders["shift_map"].get(shift.lower(), 0)

    # Encodage de l'opérateur
    le_op = encoders["le_operateur"]
    try:
        op_encoded = le_op.transform([operateur_id.lower()])[0]
    except ValueError:
        op_encoded = 0  # opérateur inconnu → valeur neutre

    # Construction du vecteur de features
    X_nouveau = pd.DataFrame([{
        "hauteur_sertissage_mm": hauteur_sertissage_mm,
        "force_arrachement_N":   force_arrachement_N,
        "resistance_ohm":        resistance_ohm,
        "temps_cycle_min":       temps_cycle_min,
        "nb_circuits":           nb_circuits,
        "nb_connecteurs":        nb_connecteurs,
        "longueur_totale_m":     longueur_totale_m,
        "shift_encoded":         shift_encoded,
        "operateur_encoded":     op_encoded,
    }])

    # Normalisation si nécessaire
    if scaled and scaler:
        X_nouveau = scaler.transform(X_nouveau)

    # Prédiction
    prediction  = model.predict(X_nouveau)[0]
    probabilite = model.predict_proba(X_nouveau)[0][1]  # P(conforme)

    # Évaluation du niveau de risque
    prob_defaut = 1 - probabilite
    if prob_defaut < 0.20:
        risque = "faible"
        recommandation = "Continuer la production normalement"
    elif prob_defaut < 0.50:
        risque = "modéré"
        recommandation = "Vérification visuelle recommandée"
    elif prob_defaut < 0.75:
        risque = "élevé"
        recommandation = "Contrôle qualité obligatoire avant expédition"
    else:
        risque = "critique"
        recommandation = "ARRÊTER — Vérifier le réglage machine immédiatement"

    return {
        "prediction":      int(prediction),
        "statut":          "conforme" if prediction == 1 else "défaut prédit",
        "probabilite_conformite": round(float(probabilite), 4),
        "probabilite_defaut":     round(float(prob_defaut), 4),
        "risque":          risque,
        "recommandation":  recommandation,
    }


# ═══════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════
def main():
    print("=" * 56)
    print("  Pipeline ML — SEWS Cabind  (Phase 4)")
    print("=" * 56)

    # 1. Features
    X, y, encoders = preparer_features()

    # 2. Split
    X_train, X_test, y_train, y_test = decouper_donnees(X, y)

    # 3. Entraînement
    modeles = entrainer_modeles(X_train, y_train)

    # 4. Évaluation
    resultats, meilleur = evaluer_modeles(modeles, X_test, y_test)

    # 5. SHAP (sur le meilleur modèle, seulement si XGBoost)
    if meilleur == "XGBoost":
        modeles[meilleur]["y_test_ref"] = y_test.values
        importance = analyser_shap(modeles[meilleur], X_train, X_test)
    else:
        print(f"\n  (SHAP disponible uniquement pour XGBoost — "
              f"meilleur modèle : {meilleur})")

    # 6. Sauvegarde
    print("\n── Sauvegarde ──────────────────────────────")
    sauvegarder_modele(modeles, meilleur, encoders)

    # 7. Test de la fonction predire
    print("\n── Test de la fonction predire ─────────────")
    test = predire(
        hauteur_sertissage_mm = 2.18,   # hors norme (cible 1.85mm)
        force_arrachement_N   = 52.0,   # faible
        resistance_ohm        = 0.038,  # élevée
        temps_cycle_min       = 48.0,
        nb_circuits           = 24,
        nb_connecteurs        = 8,
        longueur_totale_m     = 1.8,
        shift                 = "nuit",
        operateur_id          = "op_015",
    )
    print(f"\n  Faisceau test (hauteur 2.18mm, nuit, force 52N) :")
    for cle, val in test.items():
        print(f"    {cle:<35} : {val}")

    test_ok = predire(
        hauteur_sertissage_mm = 1.85,   # cible parfaite
        force_arrachement_N   = 86.0,   # normale
        resistance_ohm        = 0.012,  # normale
        temps_cycle_min       = 45.0,
        nb_circuits           = 20,
        nb_connecteurs        = 7,
        longueur_totale_m     = 1.5,
        shift                 = "matin",
        operateur_id          = "op_003",
    )
    print(f"\n  Faisceau test (hauteur 1.85mm, matin, force 86N) :")
    for cle, val in test_ok.items():
        print(f"    {cle:<35} : {val}")

    print("\n" + "=" * 56)
    print("  ✓ Phase 4 terminée — modèle sauvegardé")
    print(f"  Fichier : {MODEL_FILE}")
    print("  Prochaine étape : Phase 5 — Dashboard Streamlit")
    print("=" * 56)


if __name__ == "__main__":
    main()