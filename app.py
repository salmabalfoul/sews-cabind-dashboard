"""
Phase 5 — Dashboard Streamlit interactif
Projet : Système intelligent de pilotage des faisceaux — SEWS Cabind

Lancement : streamlit run app.py

Structure :
  - Barre latérale  : filtres globaux (date, référence, ligne, shift)
  - Onglet 1        : Tableau de bord KPI (FPY, PPM, rebut, tendances)
  - Onglet 2        : Analyse des défauts (Pareto, heatmap, opérateurs)
  - Onglet 3        : Prédiction ML (saisie paramètres → risque + recommandation)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import sys

# ── Import des modules du projet ──
sys.path.insert(0, os.path.dirname(__file__))
from kpi_engine import (
    charger_donnees, calculer_fpy, calculer_ppm,
    calculer_rebut_retravail, calculer_pareto,
    calculer_tendances, comparer_par_shift,
    comparer_par_operateur, comparer_par_reference,
    comparer_par_ligne,
)
from ml_model import predire

# ═══════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="SEWS Cabind — Pilotage Faisceaux",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé léger
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 8px;
    }
    .status-bon      { color: #28a745; font-weight: bold; }
    .status-attention{ color: #ffc107; font-weight: bold; }
    .status-critique { color: #dc3545; font-weight: bold; }
    .risque-faible   { background: #d4edda; color: #155724;
                       padding: 8px 14px; border-radius: 8px; }
    .risque-modere   { background: #fff3cd; color: #856404;
                       padding: 8px 14px; border-radius: 8px; }
    .risque-eleve    { background: #f8d7da; color: #721c24;
                       padding: 8px 14px; border-radius: 8px; }
    .risque-critique { background: #dc3545; color: white;
                       padding: 8px 14px; border-radius: 8px;
                       font-weight: bold; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# BARRE LATÉRALE — FILTRES GLOBAUX
# ═══════════════════════════════════════════
def afficher_sidebar() -> dict:
    """
    Affiche la barre de filtres et retourne les valeurs sélectionnées.
    Ces filtres s'appliquent à tous les onglets sauf la prédiction ML.
    """
    st.sidebar.image(
        "https://via.placeholder.com/200x60/1f77b4/white?text=SEWS+Cabind",
        use_column_width=True
    )
    st.sidebar.title("Filtres")
    st.sidebar.markdown("---")

    # Filtre dates
    st.sidebar.subheader("Période")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        date_debut = st.date_input("Du", value=pd.to_datetime("2023-01-01"))
    with col2:
        date_fin   = st.date_input("Au",  value=pd.to_datetime("2024-12-31"))

    # Filtre référence
    st.sidebar.subheader("Faisceau")
    references = ["Toutes", "BDX-0047", "BDX-0052", "CDX-1123", "CDX-1145",
                  "ELX-0088", "ELX-0091", "FHX-2201", "FHX-2205",
                  "GKX-3310", "GKX-3315"]
    reference = st.sidebar.selectbox("Référence", references)

    # Filtre ligne
    lignes = ["Toutes", "Ligne_A", "Ligne_B", "Ligne_C", "Ligne_D"]
    ligne  = st.sidebar.selectbox("Ligne de production", lignes)

    # Filtre shift
    st.sidebar.subheader("Équipe")
    shifts = ["Tous", "matin", "apres_midi", "nuit"]
    shift  = st.sidebar.selectbox("Shift", shifts)

    st.sidebar.markdown("---")
    st.sidebar.caption("Système intelligent de pilotage")
    st.sidebar.caption("SEWS Cabind — PFE 2024")

    return {
        "date_debut": str(date_debut),
        "date_fin":   str(date_fin),
        "reference":  reference if reference != "Toutes" else None,
        "ligne":      ligne.lower() if ligne != "Toutes" else None,
        "shift":      shift.lower() if shift != "Tous" else None,
    }


# ═══════════════════════════════════════════
# ONGLET 1 — TABLEAU DE BORD KPI
# ═══════════════════════════════════════════
def onglet_kpi(filtres: dict):
    st.header("Tableau de bord — KPI de production")

    # Chargement des données avec filtres
    with st.spinner("Calcul des KPI en cours..."):
        df = charger_donnees(**filtres)

    if len(df) == 0:
        st.warning("Aucune donnée pour ces filtres. Modifie les critères.")
        return

    # ── Ligne 1 : 4 métriques principales ──
    fpy_res = calculer_fpy(df)
    ppm_res = calculer_ppm(df)
    rr_res  = calculer_rebut_retravail(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        couleur = "normal" if fpy_res["statut"] == "bon" else \
                  "inverse" if fpy_res["statut"] == "critique" else "off"
        st.metric(
            label="FPY — First Pass Yield",
            value=f"{fpy_res['fpy']}%",
            delta=f"Objectif >80%" ,
            delta_color=couleur,
        )

    with col2:
        st.metric(
            label="PPM",
            value=f"{ppm_res['ppm']:,}",
            delta="Objectif <50 000 en proto",
            delta_color="off",
        )

    with col3:
        st.metric(
            label="Taux de rebut",
            value=f"{rr_res['taux_rebut']}%",
            delta=f"{rr_res['nb_rebut']} pièces perdues",
            delta_color="inverse",
        )

    with col4:
        st.metric(
            label="Faisceaux analysés",
            value=f"{len(df):,}",
            delta=f"Retravail : {rr_res['taux_retravail']}%",
            delta_color="off",
        )

    st.markdown("---")

    # ── Ligne 2 : Tendances hebdomadaires ──
    st.subheader("Évolution du FPY par semaine")
    tendances = calculer_tendances(df, nb_semaines=12)

    if len(tendances) > 0:
        fig_tend = go.Figure()
        fig_tend.add_trace(go.Scatter(
            x=tendances["semaine"],
            y=tendances["fpy"],
            mode="lines+markers",
            name="FPY",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        ))
        # Ligne objectif 80%
        fig_tend.add_hline(
            y=80, line_dash="dash", line_color="orange",
            annotation_text="Objectif 80%",
            annotation_position="right",
        )
        fig_tend.update_layout(
            xaxis_title="Semaine",
            yaxis_title="FPY (%)",
            yaxis=dict(range=[50, 100]),
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_tend, use_container_width=True)
    else:
        st.info("Pas assez de données pour afficher les tendances.")

    # ── Ligne 3 : Comparaison par shift ──
    st.subheader("Performance par shift")
    par_shift = comparer_par_shift(df)

    if len(par_shift) > 0:
        col_a, col_b = st.columns(2)

        with col_a:
            fig_shift_fpy = px.bar(
                par_shift,
                x="shift", y="fpy",
                color="shift",
                color_discrete_map={
                    "matin":      "#2196F3",
                    "apres_midi": "#FF9800",
                    "nuit":       "#673AB7",
                },
                title="FPY par shift (%)",
                labels={"fpy": "FPY (%)", "shift": "Shift"},
                text_auto=".1f",
            )
            fig_shift_fpy.update_layout(
                height=280, showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_shift_fpy, use_container_width=True)

        with col_b:
            fig_shift_ppm = px.bar(
                par_shift,
                x="shift", y="ppm",
                color="shift",
                color_discrete_map={
                    "matin":      "#2196F3",
                    "apres_midi": "#FF9800",
                    "nuit":       "#673AB7",
                },
                title="PPM par shift",
                labels={"ppm": "PPM", "shift": "Shift"},
                text_auto=",",
            )
            fig_shift_ppm.update_layout(
                height=280, showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_shift_ppm, use_container_width=True)

    # ── Ligne 4 : Performance par référence ──
    st.subheader("FPY par référence de faisceau")
    par_ref = comparer_par_reference(df)

    if len(par_ref) > 0:
        fig_ref = px.bar(
            par_ref.sort_values("fpy"),
            x="fpy", y="reference",
            orientation="h",
            color="fpy",
            color_continuous_scale="RdYlGn",
            range_color=[60, 100],
            title="FPY par référence (barres horizontales)",
            labels={"fpy": "FPY (%)", "reference": "Référence"},
            text_auto=".1f",
        )
        fig_ref.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ref, use_container_width=True)

    # ── Tableau récapitulatif ──
    with st.expander("Détail complet par référence"):
        st.dataframe(
            par_ref[["reference", "total", "fpy", "taux_defaut",
                      "ppm", "temps_cycle_moy"]]
            .rename(columns={
                "reference":       "Référence",
                "total":           "Total produit",
                "fpy":             "FPY (%)",
                "taux_defaut":     "Taux défaut (%)",
                "ppm":             "PPM",
                "temps_cycle_moy": "Temps cycle moy (min)",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════
# ONGLET 2 — ANALYSE DES DÉFAUTS
# ═══════════════════════════════════════════
def onglet_defauts(filtres: dict):
    st.header("Analyse qualité — Défauts et non-conformités")

    with st.spinner("Analyse en cours..."):
        df = charger_donnees(**filtres)

    if len(df) == 0:
        st.warning("Aucune donnée pour ces filtres.")
        return

    # ── Pareto des défauts ──
    st.subheader("Diagramme de Pareto — Top défauts")
    pareto = calculer_pareto(df)

    if len(pareto) > 0:
        fig_pareto = go.Figure()

        # Barres (fréquences)
        couleurs = [
            "#dc3545" if p == "haute" else "#6c757d"
            for p in pareto["priorite"]
        ]
        fig_pareto.add_trace(go.Bar(
            name="Nb de défauts",
            x=pareto["type_defaut"],
            y=pareto["nb"],
            marker_color=couleurs,
            yaxis="y",
        ))

        # Courbe cumulée (axe secondaire)
        fig_pareto.add_trace(go.Scatter(
            name="% cumulé",
            x=pareto["type_defaut"],
            y=pareto["pct_cumule"],
            mode="lines+markers",
            line=dict(color="#FF9800", width=2),
            marker=dict(size=7),
            yaxis="y2",
        ))

        # Ligne 80%
        fig_pareto.add_hline(
            y=80, line_dash="dot", line_color="#FF9800",
            annotation_text="80%", yref="y2",
        )

        fig_pareto.update_layout(
            yaxis=dict(title="Nombre de défauts"),
            yaxis2=dict(
                title="% cumulé",
                overlaying="y", side="right",
                range=[0, 105],
            ),
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # Interprétation automatique
        defauts_prioritaires = pareto[pareto["priorite"] == "haute"]["type_defaut"].tolist()
        st.info(
            f"Priorité haute ({len(defauts_prioritaires)} défaut(s) → 80% des problèmes) : "
            f"**{', '.join(defauts_prioritaires)}**. "
            f"Corriger ces défauts en premier aura le plus grand impact."
        )
    else:
        st.success("Aucun défaut détecté sur la période sélectionnée !")

    st.markdown("---")

    # ── Heatmap : défauts par ligne × shift ──
    st.subheader("Heatmap — Taux de défauts par ligne et shift")
    df_nc = df[df["statut_conformite"] == 0].copy()

    if len(df_nc) > 0:
        heatmap_data = (
            df_nc.groupby(["ligne_production", "shift"])
            .size()
            .reset_index(name="nb_defauts")
        )
        total_par_groupe = (
            df.groupby(["ligne_production", "shift"])
            .size()
            .reset_index(name="total")
        )
        heatmap_data = heatmap_data.merge(
            total_par_groupe, on=["ligne_production", "shift"]
        )
        heatmap_data["taux_pct"] = (
            heatmap_data["nb_defauts"] / heatmap_data["total"] * 100
        ).round(1)

        pivot = heatmap_data.pivot(
            index="ligne_production",
            columns="shift",
            values="taux_pct"
        ).fillna(0)

        fig_hm = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            title="Taux de défauts (%) par ligne × shift",
            labels=dict(color="Taux (%)"),
            text_auto=".1f",
        )
        fig_hm.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Performance par opérateur ──
    st.subheader("Performance par opérateur")
    par_op = comparer_par_operateur(df)

    if len(par_op) > 0:
        col_top, col_bot = st.columns(2)

        with col_top:
            st.markdown("**Top 5 meilleurs opérateurs**")
            top5 = par_op.head(5)[["operateur_id", "fpy", "total"]].copy()
            top5.columns = ["Opérateur", "FPY (%)", "Nb produits"]
            st.dataframe(top5, hide_index=True, use_container_width=True)

        with col_bot:
            st.markdown("**5 opérateurs à accompagner**")
            bot5 = par_op.tail(5)[["operateur_id", "fpy", "taux_defaut"]].copy()
            bot5.columns = ["Opérateur", "FPY (%)", "Taux défaut (%)"]
            st.dataframe(bot5, hide_index=True, use_container_width=True)

        # Graphique barres opérateurs
        fig_op = px.bar(
            par_op.sort_values("fpy"),
            x="operateur_id", y="fpy",
            color="fpy",
            color_continuous_scale="RdYlGn",
            range_color=[60, 100],
            title="FPY par opérateur",
            labels={"fpy": "FPY (%)", "operateur_id": "Opérateur"},
        )
        fig_op.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_op, use_container_width=True)

    # ── Phase de détection ──
    st.subheader("Où les défauts sont-ils détectés ?")
    if len(df_nc) > 0:
        phases = df_nc["phase_detection"].value_counts().reset_index()
        phases.columns = ["Phase", "Nb défauts"]

        fig_phases = px.pie(
            phases,
            names="Phase",
            values="Nb défauts",
            hole=0.4,
            title="Répartition par phase de détection",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_phases.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_phases, use_container_width=True)


# ═══════════════════════════════════════════
# ONGLET 3 — PRÉDICTION ML
# ═══════════════════════════════════════════
def onglet_prediction():
    st.header("Prédiction ML — Évaluation du risque qualité")
    st.markdown(
        "Entrez les mesures d'un faisceau **en cours de fabrication** "
        "pour obtenir une prédiction de risque **avant le test final**."
    )

    # Vérification que le modèle existe
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        st.error(
            "Modèle ML introuvable. Lance d'abord : `python ml_model.py`"
        )
        return

    # ── Formulaire de saisie ──
    st.subheader("Paramètres du faisceau")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Mesures physiques**")
        hauteur = st.number_input(
            "Hauteur sertissage (mm)",
            min_value=0.5, max_value=4.0,
            value=1.85, step=0.01,
            help="Cible industrie : 1.85mm ± 0.10mm",
            format="%.3f",
        )
        force = st.number_input(
            "Force d'arrachement (N)",
            min_value=5.0, max_value=200.0,
            value=85.0, step=1.0,
            help="Valeur normale : 80–90 N",
        )
        resistance = st.number_input(
            "Résistance électrique (Ω)",
            min_value=0.001, max_value=0.5,
            value=0.012, step=0.001,
            help="Valeur normale : 0.010–0.015 Ω",
            format="%.4f",
        )

    with col2:
        st.markdown("**Caractéristiques du faisceau**")
        nb_circuits = st.slider(
            "Nombre de circuits",
            min_value=5, max_value=100, value=24,
        )
        nb_connecteurs = st.slider(
            "Nombre de connecteurs",
            min_value=2, max_value=40, value=8,
        )
        longueur = st.number_input(
            "Longueur totale (m)",
            min_value=0.1, max_value=20.0,
            value=1.8, step=0.1,
        )
        temps_cycle = st.number_input(
            "Temps de cycle (min)",
            min_value=1.0, max_value=300.0,
            value=48.0, step=1.0,
        )

    with col3:
        st.markdown("**Contexte de production**")
        shift = st.selectbox(
            "Shift",
            options=["matin", "apres_midi", "nuit"],
            index=0,
        )
        operateur = st.selectbox(
            "Opérateur",
            options=[f"op_{str(i).zfill(3)}" for i in range(1, 21)],
            index=0,
        )

        # Indicateur visuel de la hauteur de sertissage
        st.markdown("**Indicateur hauteur sertissage**")
        ecart = abs(hauteur - 1.85)
        if ecart < 0.05:
            st.success(f"Dans la tolérance ({hauteur:.3f}mm)")
        elif ecart < 0.15:
            st.warning(f"Hors tolérance ({hauteur:.3f}mm, écart {ecart:.3f}mm)")
        else:
            st.error(f"Hors norme critique ({hauteur:.3f}mm, écart {ecart:.3f}mm)")

    # ── Bouton de prédiction ──
    st.markdown("---")
    if st.button("Lancer la prédiction", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            result = predire(
                hauteur_sertissage_mm=hauteur,
                force_arrachement_N=force,
                resistance_ohm=resistance,
                temps_cycle_min=temps_cycle,
                nb_circuits=nb_circuits,
                nb_connecteurs=nb_connecteurs,
                longueur_totale_m=longueur,
                shift=shift,
                operateur_id=operateur,
                chemin_modele=model_path,
            )

        # ── Affichage du résultat ──
        st.subheader("Résultat de la prédiction")
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if result["prediction"] == 1:
                st.success("CONFORME")
                st.metric("Probabilité de conformité",
                          f"{result['probabilite_conformite']*100:.1f}%")
            else:
                st.error("DÉFAUT PRÉDIT")
                st.metric("Probabilité de défaut",
                          f"{result['probabilite_defaut']*100:.1f}%")

        with col_res2:
            risque = result["risque"]
            couleurs_risque = {
                "faible":   "status-bon",
                "modéré":   "status-attention",
                "élevé":    "status-critique",
                "critique": "status-critique",
            }
            st.markdown(f"**Niveau de risque**")
            classe = couleurs_risque.get(risque, "status-attention")
            st.markdown(
                f'<span class="{classe}" style="font-size:1.4em">'
                f'{risque.upper()}</span>',
                unsafe_allow_html=True,
            )

        with col_res3:
            st.markdown("**Recommandation**")
            st.info(result["recommandation"])

        # ── Jauge de probabilité ──
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result["probabilite_conformite"] * 100,
            number={"suffix": "%"},
            title={"text": "Probabilité de conformité"},
            delta={"reference": 80, "increasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#1f77b4"},
                "steps": [
                    {"range": [0,  50], "color": "#dc3545"},
                    {"range": [50, 80], "color": "#ffc107"},
                    {"range": [80, 100],"color": "#28a745"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=0),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Détail des valeurs saisies ──
        with st.expander("Détail des paramètres saisis"):
            recap = pd.DataFrame([{
                "Paramètre": "Hauteur sertissage",
                "Valeur":    f"{hauteur:.3f} mm",
                "Référence": "1.85 ± 0.10 mm",
            }, {
                "Paramètre": "Force d'arrachement",
                "Valeur":    f"{force:.1f} N",
                "Référence": "80–90 N",
            }, {
                "Paramètre": "Résistance électrique",
                "Valeur":    f"{resistance:.4f} Ω",
                "Référence": "0.010–0.015 Ω",
            }, {
                "Paramètre": "Nb circuits",
                "Valeur":    str(nb_circuits),
                "Référence": "—",
            }, {
                "Paramètre": "Nb connecteurs",
                "Valeur":    str(nb_connecteurs),
                "Référence": "—",
            }, {
                "Paramètre": "Shift",
                "Valeur":    shift,
                "Référence": "matin = moins de risques",
            }, {
                "Paramètre": "Opérateur",
                "Valeur":    operateur,
                "Référence": "—",
            }])
            st.dataframe(recap, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════
def main():
    # En-tête principal
    st.title("Système intelligent de pilotage des faisceaux prototypes")
    st.markdown(
        "**SEWS Cabind Maroc** — Tableau de bord qualité et prédiction ML"
    )
    st.markdown("---")

    # Filtres sidebar
    filtres = afficher_sidebar()

    # Onglets
    tab1, tab2, tab3 = st.tabs([
        "Tableau de bord KPI",
        "Analyse des défauts",
        "Prédiction ML",
    ])

    with tab1:
        onglet_kpi(filtres)

    with tab2:
        onglet_defauts(filtres)

    with tab3:
        onglet_prediction()


if __name__ == "__main__":
    main()