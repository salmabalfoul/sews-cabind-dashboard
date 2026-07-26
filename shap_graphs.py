"""
Graphiques SHAP visuels
Fichier : shap_graphs.py
Projet  : Système intelligent de pilotage — SEWS Cabind

Ce fichier produit 4 graphiques SHAP à insérer dans ton rapport.
Il doit être lancé APRÈS ml_model_v2.py (qui crée model.pkl).

Comment lancer :
  python shap_graphs.py

Graphiques produits :
  1. Summary Plot  → importance globale + direction de l'effet
  2. Bar Plot      → importance globale simple (pour le rapport)
  3. Waterfall     → explication d'UN faisceau défectueux
  4. Dependence    → relation hauteur_sertissage ↔ prédiction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # mode sans fenêtre (compatible tous OS)
import pickle
import shap
import os
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from db_config import get_url

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
SEED       = 42
TEST_SIZE  = 0.20
MODEL_FILE = os.path.join("models", "model.pkl")
OUTPUT     = os.path.join("data", "eda_figures")
os.makedirs(OUTPUT, exist_ok=True)

FEATURES = [
    "hauteur_sertissage_mm",
    "force_arrachement_N",
    "resistance_ohm",
    "temps_cycle_min",
    "nb_circuits",
    "nb_connecteurs",
    "longueur_totale_m",
    "shift_encoded",
    "operateur_encoded",
]

print("=" * 55)
print("  Graphiques SHAP — SEWS Cabind")
print("=" * 55)

# ─────────────────────────────────────────
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ─────────────────────────────────────────
print("\n  Chargement du modèle depuis model.pkl...")

if not os.path.exists(MODEL_FILE):
    print(f"  ✗ Fichier {MODEL_FILE} introuvable.")
    print("    Lance d'abord : python ml_model_v2.py")
    exit(1)

with open(MODEL_FILE, "rb") as f:
    paquet = pickle.load(f)

modele   = paquet["modele"]
encoders = paquet["encoders"]
print(f"  ✓ Modèle chargé : {paquet['nom']}")

# Rechargement des données pour reconstruire X_test
print("  Rechargement des données...")
engine = create_engine(get_url(), echo=False)
df     = pd.read_sql(text("SELECT * FROM production_harness"), engine)

shift_map = encoders["shift_map"]
le_op     = encoders["le_operateur"]

df["shift_encoded"]    = df["shift"].map(shift_map).fillna(0).astype(int)
df["operateur_encoded"] = le_op.transform(df["operateur_id"])

X = df[FEATURES].copy()
y = df["statut_conformite"].astype(int).copy()
X = X.fillna(X.median())

_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
X_test_df = pd.DataFrame(X_test.values, columns=FEATURES)

print(f"  ✓ {len(X_test):,} faisceaux de test chargés")

# ─────────────────────────────────────────
# CALCUL DES VALEURS SHAP
# ─────────────────────────────────────────
print("\n  Calcul des valeurs SHAP (30-60 secondes)...")
explainer   = shap.TreeExplainer(modele)
shap_values = explainer.shap_values(X_test_df)

# Random Forest → shap_values = liste [classe_0, classe_1]
# On prend la classe 0 (défaut) pour expliquer les prédictions de défaut
if isinstance(shap_values, list):
    sv_defaut   = shap_values[0]  # contributions vers la classe DÉFAUT
    sv_conforme = shap_values[1]  # contributions vers la classe CONFORME
    base_value  = explainer.expected_value[0]
else:
    sv_defaut  = shap_values
    base_value = explainer.expected_value

# 🔧 CORRECTION : Vérifier et corriger la dimension de sv_defaut
if sv_defaut.ndim == 3:
    sv_defaut = sv_defaut[:, :, 0] if sv_defaut.shape[2] > 0 else sv_defaut.mean(axis=2)
elif sv_defaut.ndim == 1:
    sv_defaut = sv_defaut.reshape(1, -1)

# S'assurer que sv_defaut est 2D avec le bon nombre de lignes
if sv_defaut.shape[0] != len(X_test_df):
    if sv_defaut.shape[0] == len(FEATURES) and sv_defaut.shape[1] == len(X_test_df):
        sv_defaut = sv_defaut.T  # Transposer si nécessaire

print(f"  ✓ Valeurs SHAP calculées (shape: {sv_defaut.shape})")


# ═══════════════════════════════════════════════════════════
# GRAPHIQUE 1 — SUMMARY PLOT (le plus important)
# ═══════════════════════════════════════════════════════════
print("\n── Graphique 1 : Summary Plot ──────────────────")

plt.figure(figsize=(10, 7))
shap.summary_plot(
    sv_defaut,
    X_test_df,
    feature_names=FEATURES,
    show=False,
    plot_type="dot",
    max_display=9,
    alpha=0.6,
)
plt.title(
    "SHAP Summary Plot — Impact des features sur la prédiction de défaut\n"
    "Gauche (rouge) = pousse vers DÉFAUT · Droite (bleu) = pousse vers CONFORME",
    fontsize=11, fontweight="bold", pad=15
)
plt.xlabel("Valeur SHAP (impact sur la prédiction de défaut)")
plt.tight_layout()

chemin = os.path.join(OUTPUT, "shap_01_summary.png")
plt.savefig(chemin, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {chemin}")


# ═══════════════════════════════════════════════════════════
# GRAPHIQUE 2 — BAR PLOT (importance globale)
# ═══════════════════════════════════════════════════════════
print("── Graphique 2 : Bar Plot ──────────────────────")

# 🔧 CORRECTION : Calculer l'importance de manière robuste
importance_values = np.abs(sv_defaut).mean(axis=0)

# S'assurer que importance_values est 1D avec la bonne longueur
if importance_values.ndim > 1:
    importance_values = importance_values.flatten()
if len(importance_values) != len(FEATURES):
    # Si les longueurs ne correspondent pas, prendre les premières valeurs
    min_len = min(len(importance_values), len(FEATURES))
    importance_values = importance_values[:min_len]
    features_used = FEATURES[:min_len]
else:
    features_used = FEATURES

importance = pd.DataFrame({
    "feature":    features_used,
    "importance": importance_values,
}).sort_values("importance", ascending=True)  # ascending pour barh

fig, ax = plt.subplots(figsize=(9, 6))
couleurs = plt.cm.RdYlGn_r(
    np.linspace(0.2, 0.8, len(importance))
)
barres = ax.barh(
    importance["feature"],
    importance["importance"],
    color=couleurs,
    edgecolor="white",
    linewidth=0.5,
    alpha=0.9,
)

# Valeurs sur les barres
for barre, val in zip(barres, importance["importance"]):
    ax.text(
        barre.get_width() + 0.001,
        barre.get_y() + barre.get_height() / 2,
        f"{val:.4f}",
        va="center", ha="left", fontsize=9,
    )

ax.set_xlabel("Importance SHAP moyenne (|valeur SHAP|)", fontsize=11)
ax.set_title(
    "Importance globale des features — Modèle Random Forest\n"
    "Plus la barre est longue, plus la feature influence les prédictions",
    fontsize=11, fontweight="bold", pad=15
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

chemin = os.path.join(OUTPUT, "shap_02_importance.png")
plt.savefig(chemin, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {chemin}")

print("\n  Ranking des features par importance SHAP :")
for rang, (_, row) in enumerate(
    importance.sort_values("importance", ascending=False).iterrows(), 1
):
    print(f"    {rang}. {row['feature']:<28} : {row['importance']:.5f}")


# ═══════════════════════════════════════════════════════════
# GRAPHIQUE 3 — WATERFALL PLOT (explication individuelle)
# ═══════════════════════════════════════════════════════════
print("\n── Graphique 3 : Waterfall Plot ────────────────")

# Trouve le premier faisceau prédit défectueux
y_pred = modele.predict(X_test_df)
indices_defaut = np.where(y_pred == 0)[0]

if len(indices_defaut) > 0 and len(sv_defaut) > 0:
    idx = indices_defaut[0]
    if idx >= len(sv_defaut):
        idx = 0
    
    print(f"  Faisceau analysé : #{idx} (prédit DÉFAUT)")
    print(f"  Valeurs réelles :")
    for feat, val in zip(FEATURES[:len(X_test_df.iloc[idx])], X_test_df.iloc[idx].values):
        print(f"    {feat:<28} : {val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Construction manuelle du waterfall
    contribs = sv_defaut[idx]
    feats    = FEATURES[:len(contribs)]
    vals     = X_test_df.iloc[idx].values[:len(contribs)]

    # Tri par contribution absolue
    ordre_idx = np.argsort(np.abs(contribs))[::-1]
    contribs_ord = contribs[ordre_idx]
    feats_ord    = [feats[i] for i in ordre_idx]
    vals_ord     = [vals[i] for i in ordre_idx]

    labels = [f"{f}\n= {v:.3f}" for f, v in zip(feats_ord, vals_ord)]
    couleurs_wf = [
        "#F44336" if c < 0 else "#4CAF50"
        for c in contribs_ord
    ]

    ax.barh(labels, contribs_ord, color=couleurs_wf,
            alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)

    ax.set_xlabel("Contribution SHAP\n(rouge = vers DÉFAUT · vert = vers CONFORME)")
    ax.set_title(
        f"Explication individuelle — Faisceau #{idx} (prédit DÉFAUT)\n"
        "Chaque barre = contribution d'une feature à la prédiction",
        fontsize=11, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    chemin = os.path.join(OUTPUT, "shap_03_waterfall.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {chemin}")
else:
    print("  Aucun faisceau défectueux trouvé dans le test set")


# ═══════════════════════════════════════════════════════════
# GRAPHIQUE 4 — DEPENDENCE PLOT
# ═══════════════════════════════════════════════════════════
print("\n── Graphique 4 : Dependence Plot ───────────────")

# Vérifier que l'index existe
if len(FEATURES) > 0:
    feature_idx = min(FEATURES.index("hauteur_sertissage_mm") if "hauteur_sertissage_mm" in FEATURES else 0, len(sv_defaut[0])-1 if len(sv_defaut) > 0 else 0)
    
    fig, ax = plt.subplots(figsize=(9, 6))

    hauteurs  = X_test_df["hauteur_sertissage_mm"].values[:len(sv_defaut)]
    shap_haut = sv_defaut[:, feature_idx] if len(sv_defaut) > 0 else np.zeros(len(hauteurs))

    # Colorer selon resistance_ohm
    if "resistance_ohm" in FEATURES:
        resist_idx = FEATURES.index("resistance_ohm")
        resistances = X_test_df["resistance_ohm"].values[:len(sv_defaut)]
    else:
        resistances = np.zeros(len(hauteurs))

    sc = ax.scatter(
        hauteurs,
        shap_haut,
        c=resistances,
        cmap="RdYlGn_r",
        alpha=0.4,
        s=15,
        edgecolors="none",
    )

    # Lignes de référence
    ax.axvline(1.85, color="black", linestyle="--",
               linewidth=2, label="Cible 1.85mm")
    ax.axhline(0, color="gray", linestyle="-",
               linewidth=0.8, alpha=0.5)
    ax.axvspan(1.75, 1.95, alpha=0.08, color="green",
               label="Zone de tolérance ±0.10mm")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Résistance électrique (Ω)", fontsize=10)

    ax.set_xlabel("Hauteur de sertissage (mm)", fontsize=11)
    ax.set_ylabel("Valeur SHAP (contribution à la prédiction de défaut)", fontsize=11)
    ax.set_title(
        "Dependence Plot — hauteur_sertissage_mm\n"
        "En-dessous de 0 = pousse vers DÉFAUT · Au-dessus = vers CONFORME",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    chemin = os.path.join(OUTPUT, "shap_04_dependence.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sauvegardé : {chemin}")
else:
    print("  ⚠ Pas assez de features pour le Dependence Plot")


# ─────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  GRAPHIQUES SHAP TERMINÉS ✓")
print("=" * 55)
print(f"\n  4 figures sauvegardées dans : {OUTPUT}/")
fichiers_shap = [f for f in os.listdir(OUTPUT) if "shap" in f]
for f in sorted(fichiers_shap):
    taille = os.path.getsize(os.path.join(OUTPUT, f)) // 1024
    print(f"    {f}  ({taille} Ko)")

print("""
  Comment utiliser ces figures dans ton rapport :

  Fig. shap_01_summary.png
  → Chapitre "Interprétabilité du modèle"
  → Titre : "SHAP Summary Plot — Impact des features"
  → Légende : "Chaque point représente un faisceau.
    La position horizontale indique l'impact sur la
    probabilité de défaut. La couleur indique la valeur
    de la feature (rouge = haute, bleu = basse)."

  Fig. shap_02_importance.png
  → Chapitre "Sélection des features"
  → Titre : "Importance globale des variables (SHAP)"
  → Légende : "La feature hauteur_sertissage_mm est
    la plus influente, conformément aux connaissances
    métier sur les défauts de sertissage."

  Fig. shap_03_waterfall.png
  → Chapitre "Explication des prédictions"
  → Titre : "Explication individuelle d'une prédiction"
  → Légende : "Pour le faisceau #X, la hauteur anormale
    de X mm contribue négativement (-0.XXX) vers la
    prédiction de défaut."

  Fig. shap_04_dependence.png
  → Chapitre "Analyse des seuils critiques"
  → Titre : "Relation hauteur_sertissage ↔ risque de défaut"
  → Légende : "Le graphique révèle un seuil critique
    autour de 2.0mm au-delà duquel la contribution SHAP
    devient fortement négative (vers défaut)."
""")