"""
Dashboard Encadrant — SEWS Cabind Prototype
Version finale corrigée — mise à jour automatique stable
Lancement : streamlit run app_encadrant.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from io import BytesIO
from datetime import datetime
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")

DB_PATH = os.path.join("data", "sews_reel.db")

FICHIERS_SURVEILLES = [
    "Employee_retard_schedule.xlsx",
    "Proto_team_affectation_dmx.xlsx",
    "Temps_standard_PROTO.xlsx",
    "PMSA_5803203185_00_23AZ013.xlsx",
    "situation_de_la_coupe_wk24.xlsx",
    "Classeur3.xlsx",
]

SEWS_BLEU       = "#003DA5"
SEWS_BLEU_LIGHT = "#0066CC"
SEWS_VERT       = "#00A651"
SEWS_ORANGE     = "#F7941D"
SEWS_ROUGE      = "#ED1C24"
SEWS_GRIS       = "#F4F6F9"
SEWS_GRIS_DARK  = "#6C757D"

st.set_page_config(
    page_title="SEWS Cabind — Pilotage Prototype",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {SEWS_GRIS}; }}
    .header-sews {{
        background: linear-gradient(135deg, {SEWS_BLEU} 0%, {SEWS_BLEU_LIGHT} 100%);
        padding: 18px 28px; border-radius: 12px;
        display: flex; align-items: center; gap: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,61,165,0.3);
    }}
    .header-titre {{
        color: white; font-size: 24px; font-weight: bold; margin: 0;
    }}
    .header-sous {{
        color: rgba(255,255,255,0.85); font-size: 13px; margin: 4px 0 0 0;
    }}
    div[data-testid="metric-container"] {{
        background: white; border-radius: 10px; padding: 14px 18px;
        border-top: 4px solid {SEWS_BLEU};
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    div[data-testid="metric-container"] label {{
        color: {SEWS_GRIS_DARK} !important; font-size: 13px !important;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {SEWS_BLEU} !important; font-size: 28px !important;
        font-weight: bold !important;
    }}
    .alerte-rouge {{
        background: {SEWS_ROUGE}; color: white;
        padding: 12px 18px; border-radius: 8px;
        font-weight: bold; margin: 6px 0; font-size: 14px;
        border-left: 5px solid #8B0000;
    }}
    .alerte-orange {{
        background: {SEWS_ORANGE}; color: white;
        padding: 12px 18px; border-radius: 8px;
        font-weight: bold; margin: 6px 0; font-size: 14px;
    }}
    [data-testid="stSidebar"] {{
        background: white; border-right: 2px solid {SEWS_BLEU};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: white; border-radius: 8px; padding: 4px; gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px; font-weight: 600; color: {SEWS_BLEU};
    }}
    .stTabs [aria-selected="true"] {{
        background: {SEWS_BLEU} !important; color: white !important;
    }}
    h2 {{ color: {SEWS_BLEU} !important;
          border-bottom: 2px solid {SEWS_BLEU}; padding-bottom: 6px; }}
    h3 {{ color: {SEWS_BLEU_LIGHT} !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MISE À JOUR AUTOMATIQUE — VERSION CORRIGÉE
# ═══════════════════════════════════════════════════════════
def get_dates_modification():
    """Retourne les dates de modification des fichiers Excel."""
    return {
        f: os.path.getmtime(f)
        for f in FICHIERS_SURVEILLES
        if os.path.exists(f)
    }


def initialiser_surveillance():
    """
    Initialise la surveillance SANS relancer l'ETL.
    Appelé uniquement au premier chargement.
    """
    if "dates_fichiers" not in st.session_state:
        st.session_state["dates_fichiers"] = get_dates_modification()
        st.session_state["derniere_maj"] = datetime.now().strftime(
            "%d/%m/%Y à %H:%M:%S"
        )
        st.session_state["etl_initial_fait"] = True


def verifier_changements():
    """
    Vérifie si un fichier Excel a été modifié DEPUIS la dernière
    lecture. Relance l'ETL seulement si vraiment modifié.
    """
    dates_actuelles   = get_dates_modification()
    dates_precedentes = st.session_state.get("dates_fichiers", {})

    fichiers_modifies = [
        f for f, d in dates_actuelles.items()
        if d != dates_precedentes.get(f)
        and f in dates_precedentes  # ignoré si premier chargement
    ]

    if fichiers_modifies and st.session_state.get("etl_initial_fait", False):
        noms = [os.path.basename(f) for f in fichiers_modifies]
        with st.spinner(f"🔄 Fichier modifié détecté : {', '.join(noms)} — Mise à jour..."):
            try:
                result = subprocess.run(
                    [sys.executable, "etl_reel.py"],
                    capture_output=True, text=True, timeout=90
                )
                if result.returncode == 0:
                    st.session_state["dates_fichiers"] = dates_actuelles
                    st.session_state["derniere_maj"] = datetime.now().strftime(
                        "%d/%m/%Y à %H:%M:%S"
                    )
                    st.cache_data.clear()
                    st.success(
                        f"✅ Données mises à jour automatiquement — "
                        f"{st.session_state['derniere_maj']}"
                    )
                    st.rerun()
                else:
                    st.warning(
                        "⚠ Mise à jour partielle — "
                        "Cliquez sur 'Forcer la mise à jour' si besoin"
                    )
            except subprocess.TimeoutExpired:
                st.warning("⚠ Mise à jour trop longue — réessayez manuellement")
            except Exception as e:
                st.warning(f"⚠ Erreur de mise à jour : {e}")


def forcer_mise_a_jour():
    """Relance l'ETL manuellement (bouton sidebar)."""
    try:
        result = subprocess.run(
            [sys.executable, "etl_reel.py"],
            capture_output=True, text=True, timeout=90
        )
        if result.returncode == 0:
            st.session_state["dates_fichiers"] = get_dates_modification()
            st.session_state["derniere_maj"] = datetime.now().strftime(
                "%d/%m/%Y à %H:%M:%S"
            )
            st.cache_data.clear()
            return True, "Mise à jour réussie"
        else:
            return False, result.stderr[:200]
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def charger_tout():
    if not os.path.exists(DB_PATH):
        return {}
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    tables = {}
    for nom in ["employes","retards","affectations","temps_standards",
                "suivi_coupe","ordres_komax","manhours_2025","anomalies_pmsa"]:
        try:
            tables[nom] = pd.read_sql(text(f"SELECT * FROM [{nom}]"), engine)
        except:
            tables[nom] = pd.DataFrame()
    return tables


def preparer_operateurs(tables):
    df_emp = tables.get("employes",     pd.DataFrame()).copy()
    df_ret = tables.get("retards",      pd.DataFrame()).copy()
    df_aff = tables.get("affectations", pd.DataFrame()).copy()
    if df_emp.empty:
        return pd.DataFrame()
    df_emp["nom_clean"] = df_emp["nom_prenom"].str.lower().str.strip()
    df_ret["nom_clean"] = df_ret["nom_prenom"].str.lower().str.strip()
    df_aff["nom_clean"] = df_aff["nom_prenom"].str.lower().str.strip()
    df = df_emp.merge(
        df_ret[["nom_clean","nb_jours_retard"]].drop_duplicates("nom_clean"),
        on="nom_clean", how="left"
    ).fillna({"nb_jours_retard": 0})
    df = df.merge(
        df_aff[["nom_clean","zone","tps_std_h","reference_spn"]]
        .drop_duplicates("nom_clean"),
        on="nom_clean", how="left"
    )
    df["nb_jours_retard"] = df["nb_jours_retard"].astype(int)
    df["zone"] = df["zone"].fillna("Non affecté")
    df["statut_retard"] = df["nb_jours_retard"].apply(
        lambda n: "🟢 Ponctuel" if n==0
        else "🟡 Quelques retards" if n<=3
        else "🟠 Retards fréquents" if n<=10
        else "🔴 Retards critiques"
    )
    df["statut_code"] = df["nb_jours_retard"].apply(
        lambda n: "Ponctuel" if n==0
        else "Quelques retards" if n<=3
        else "Retards fréquents" if n<=10
        else "Retards critiques"
    )
    return df


# ─────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────
def afficher_entete():
    logo = """
    <svg width="70" height="36" viewBox="0 0 70 36" xmlns="http://www.w3.org/2000/svg">
      <rect width="70" height="36" rx="5" fill="rgba(255,255,255,0.18)"/>
      <text x="35" y="24" text-anchor="middle" font-family="Arial Black,sans-serif"
            font-size="18" font-weight="900" fill="white" letter-spacing="2">SEWS</text>
    </svg>"""
    st.markdown(f"""
    <div class="header-sews">
        <div>{logo}</div>
        <div>
            <div class="header-titre">
                ⚡ SEWS Cabind Maroc — Pilotage Prototype</div>
            <div class="header-sous">
                Tableau de bord opérationnel · Équipe Prototype MY2027 ·
                {datetime.now().strftime('%A %d %B %Y, %H:%M')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR PROPRE
# ─────────────────────────────────────────────────────────
def afficher_sidebar(tables):
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0 10px;">
            <div style="font-size:40px;">⚡</div>
            <div style="color:{SEWS_BLEU};font-weight:bold;font-size:18px;
                        margin-top:4px;">SEWS Cabind</div>
            <div style="color:{SEWS_GRIS_DARK};font-size:12px;">
                Pilotage Prototype MY2027</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Date et heure
        st.markdown(f"""
        <div style="background:{SEWS_GRIS};border-radius:8px;
                    padding:10px 14px;margin-bottom:12px;">
            <div style="color:{SEWS_BLEU};font-weight:bold;font-size:13px;">
                📅 {datetime.now().strftime('%d/%m/%Y')}</div>
            <div style="color:{SEWS_GRIS_DARK};font-size:12px;">
                🕐 {datetime.now().strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Statut des fichiers
        st.markdown(f"""
        <div style="color:{SEWS_BLEU};font-weight:bold;font-size:14px;
                    margin-bottom:8px;">📂 Sources de données</div>
        """, unsafe_allow_html=True)

        for fichier in FICHIERS_SURVEILLES:
            existe = os.path.exists(fichier)
            nom_court = fichier.replace(".xlsx","")[:24]
            if existe:
                mtime = datetime.fromtimestamp(
                    os.path.getmtime(fichier)
                ).strftime("%d/%m %H:%M")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            align-items:center;padding:4px 0;
                            border-bottom:1px solid #f0f0f0;">
                    <span style="font-size:11px;color:#333;">✅ {nom_court}</span>
                    <span style="font-size:10px;color:{SEWS_GRIS_DARK};">{mtime}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding:4px 0;border-bottom:1px solid #f0f0f0;">
                    <span style="font-size:11px;color:{SEWS_ROUGE};">
                        ❌ {nom_court}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Statut mise à jour auto
        derniere_maj = st.session_state.get(
            "derniere_maj",
            datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        )
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{SEWS_BLEU},{SEWS_BLEU_LIGHT});
                    border-radius:8px;padding:12px;margin-bottom:8px;color:white;">
            <div style="font-weight:bold;font-size:13px;margin-bottom:4px;">
                🔄 Mise à jour automatique</div>
            <div style="font-size:11px;opacity:0.9;">Active · vérif. 30 sec</div>
            <div style="font-size:10px;opacity:0.8;margin-top:4px;">
                Modifiez un Excel → sauvegardez<br>→ dashboard mis à jour auto</div>
        </div>
        <div style="font-size:10px;color:{SEWS_GRIS_DARK};
                    text-align:center;margin-bottom:10px;">
            Dernière MAJ : <b>{derniere_maj}</b>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Forcer la mise à jour",
                     use_container_width=True, type="primary"):
            with st.spinner("Mise à jour..."):
                ok, msg = forcer_mise_a_jour()
            if ok:
                st.success("✅ Fait !")
                st.rerun()
            else:
                st.error(f"Erreur : {msg}")

        st.markdown("---")

        # KPIs rapides
        df_op    = preparer_operateurs(tables)
        df_coupe = tables.get("suivi_coupe",   pd.DataFrame())
        df_anom  = tables.get("anomalies_pmsa",pd.DataFrame())

        if not df_op.empty:
            nb_op   = len(df_op)
            nb_ret  = (df_op["nb_jours_retard"]>0).sum()
            nb_bloq = int((df_coupe["pct_coupe"]==0).sum()) if not df_coupe.empty else 0
            nb_an   = len(df_anom) if not df_anom.empty else 0

            st.markdown(f"""
            <div style="color:{SEWS_BLEU};font-weight:bold;
                        font-size:14px;margin-bottom:8px;">
                📊 Résumé rapide</div>
            """, unsafe_allow_html=True)

            for label, val, couleur in [
                ("👷 Opérateurs",       nb_op,  SEWS_BLEU),
                ("🟠 Avec retards",     nb_ret, SEWS_ORANGE if nb_ret>0 else SEWS_VERT),
                ("🔴 Faisceaux bloqués",nb_bloq,SEWS_ROUGE  if nb_bloq>0 else SEWS_VERT),
                ("📨 Anomalies Italie", nb_an,  SEWS_ORANGE if nb_an>0  else SEWS_VERT),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            align-items:center;padding:6px 10px;
                            background:{SEWS_GRIS};border-radius:6px;
                            margin-bottom:4px;">
                    <span style="font-size:12px;color:#333;">{label}</span>
                    <span style="font-weight:bold;color:{couleur};
                                 font-size:16px;">{val}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "PFE Master Data Science\n"
            "ENS Martil · SEWS Cabind 2025"
        )


# ═══════════════════════════════════════════════════════════
# ONGLET 1 — OPÉRATEURS
# ═══════════════════════════════════════════════════════════
def onglet_operateurs(tables):
    st.header("👷 Opérateurs — Présence & Ponctualité")
    df = preparer_operateurs(tables)
    if df.empty:
        st.error("Aucune donnée. Vérifiez les fichiers Excel.")
        return

    total  = len(df)
    nb_p   = (df["nb_jours_retard"]==0).sum()
    nb_r   = (df["nb_jours_retard"]>0).sum()
    total_j= int(df["nb_jours_retard"].sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👷 Total opérateurs",   str(total))
    c2.metric("🟢 Ponctuels",          str(nb_p),   f"{nb_p/total*100:.0f}%")
    c3.metric("🟠 Avec retards",       str(nb_r),   f"{nb_r} opérateurs")
    c4.metric("📅 Total jours retard", str(total_j),"cumulé 2025")

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        zones  = ["Toutes"] + sorted(df["zone"].dropna().unique().tolist())
        zone_s = st.selectbox("🔍 Zone", zones)
    with col2:
        statuts= ["Tous","Ponctuel","Quelques retards",
                  "Retards fréquents","Retards critiques"]
        stat_s = st.selectbox("🚦 Statut", statuts)

    df_f = df.copy()
    if zone_s  != "Toutes": df_f = df_f[df_f["zone"]        == zone_s]
    if stat_s  != "Tous":   df_f = df_f[df_f["statut_code"] == stat_s]

    st.subheader(f"Liste des opérateurs ({len(df_f)})")
    df_tab = df_f[["nom_prenom","zone","nb_jours_retard",
                   "statut_retard","reference_spn"]].copy()
    df_tab.columns = ["Opérateur","Zone","Jours retard","Statut","Faisceau"]
    st.dataframe(df_tab.sort_values("Jours retard",ascending=False),
                 use_container_width=True, hide_index=True, height=460)

    st.markdown("---")
    rz = df.groupby("zone")["nb_jours_retard"].sum().reset_index()
    rz = rz[rz["nb_jours_retard"]>0].sort_values("nb_jours_retard",ascending=False)
    if not rz.empty:
        st.subheader("📊 Retards par zone")
        fig = px.bar(rz, x="zone", y="nb_jours_retard",
                     color="nb_jours_retard",
                     color_continuous_scale=[SEWS_VERT,SEWS_ORANGE,SEWS_ROUGE],
                     title="Jours de retard cumulés par zone",
                     labels={"zone":"Zone","nb_jours_retard":"Jours"},
                     text_auto=True)
        fig.update_layout(height=300, coloraxis_showscale=False,
                          margin=dict(t=40,b=0,l=0,r=0),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    df_sv = df[df["nb_jours_retard"]>0].sort_values(
        "nb_jours_retard",ascending=False
    )[["nom_prenom","zone","nb_jours_retard","statut_retard"]]
    if not df_sv.empty:
        st.subheader("⚠ Opérateurs à suivre")
        df_sv.columns = ["Opérateur","Zone","Jours retard","Statut"]
        st.dataframe(df_sv, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# ONGLET 2 — PERFORMANCE
# ═══════════════════════════════════════════════════════════
def onglet_performance(tables):
    st.header("📊 Performance — Zones & Classement")
    df_op  = preparer_operateurs(tables)
    df_aff = tables.get("affectations",   pd.DataFrame()).copy()
    df_tps = tables.get("temps_standards",pd.DataFrame()).copy()
    df_mh  = tables.get("manhours_2025",  pd.DataFrame()).copy()
    if df_op.empty:
        st.error("Données non disponibles.")
        return

    st.subheader("📋 Répartition par zone")
    zone_fam = {"CABINA":"CABINA","Engine":"Engine",
                "BRIGLIA UREA":"BRIGLIA UREA","COFANO":"COFANO","COFANO NDE":"COFANO"}
    tps_d = dict(zip(df_tps["famille"],df_tps["tps_proto_h"])) if not df_tps.empty else {}
    cad_d = dict(zip(df_tps["famille"],df_tps["cadence_txt"])) if not df_tps.empty else {}

    zones_data = []
    for zone, grp in df_aff.groupby("zone"):
        fam = zone_fam.get(zone,"—")
        tps = tps_d.get(fam)
        zones_data.append({
            "Zone":          zone,
            "Nb opérateurs": len(grp),
            "Temps standard":f"{tps:.2f}h/pièce" if tps else "variable",
            "Cadence":       cad_d.get(fam,"variable") or "variable",
            "Opérateurs":    ", ".join(grp["nom_prenom"].tolist()),
        })
    df_zones = pd.DataFrame(zones_data)
    st.dataframe(df_zones, use_container_width=True,
                 hide_index=True, height=380)

    fig_pie = px.pie(df_zones, names="Zone", values="Nb opérateurs",
                     title="Opérateurs par zone",
                     color_discrete_sequence=[
                         SEWS_BLEU,SEWS_BLEU_LIGHT,SEWS_VERT,SEWS_ORANGE,
                         "#9DC3E6","#70AD47","#FF6B6B","#A5A5A5",
                         "#FFC000","#4472C4","#ED7D31","#5A5A5A","#7030A0"],
                     hole=0.38)
    fig_pie.update_layout(height=340, margin=dict(t=50,b=0,l=0,r=0),
                          paper_bgcolor="white")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Classement des opérateurs")
    taux_zone = {
        "BRIGLIA UREA":20,"Engine":16,"CABINA":14,
        "COFANO":8,"COFANO NDE":8,"PREMONTAGE":18,
        "CONTRÔLE ELECTRIQUE+PIN TO PIN":6,"CONTRÔLE FINAL":10,
        "SERTISSAGE +épissurage":6,"PREPARATION GAINE":8,
        "épissurage":8,"support PM":10,
        "PREPARATION ET VALIDATION LES TABLES DE MONTAGE":10,
    }
    rows_cl = []
    for _, row in df_op.iterrows():
        zone = str(row.get("zone","")).strip()
        tb   = taux_zone.get(zone,2)
        nb_r = int(row["nb_jours_retard"])
        if   nb_r>=10: taux=max(2,tb-14)
        elif nb_r>=5:  taux=max(2,tb-8)
        elif nb_r>=3:  taux=max(2,tb-4)
        elif nb_r>=1:  taux=max(2,tb-2)
        else:          taux=tb
        rows_cl.append({
            "Opérateur":row["nom_prenom"],"Zone":zone,
            "Taux":taux,"Score":min(100,taux*5),"Retards":nb_r,
        })
    df_cl = pd.DataFrame(rows_cl).sort_values("Score",ascending=False)
    df_cl.index = range(1,len(df_cl)+1)
    medals = {1:"🥇",2:"🥈",3:"🥉"}
    df_cl["Rang"]   = [medals.get(i,str(i)) for i in df_cl.index]
    df_cl["Taux %"] = df_cl["Taux"].apply(lambda t:f"+{t}%")
    df_cl["Score /100"] = df_cl["Score"]
    st.dataframe(df_cl[["Rang","Opérateur","Zone","Taux %","Score /100","Retards"]],
                 use_container_width=True, hide_index=True, height=460)

    top20 = df_cl.head(20).reset_index(drop=True)
    fig_cl = go.Figure(go.Bar(
        x=top20["Score /100"], y=top20["Opérateur"], orientation="h",
        marker_color=[SEWS_VERT if s>=80 else SEWS_ORANGE if s>=50 else SEWS_ROUGE
                      for s in top20["Score /100"]],
        text=top20["Score /100"].apply(lambda s:f"{s}/100"),
        textposition="outside",
    ))
    fig_cl.add_vline(x=80,line_dash="dash",line_color=SEWS_BLEU,
                     annotation_text="Objectif 80")
    fig_cl.update_layout(
        title="Score de performance (Top 20)",
        xaxis=dict(range=[0,115],title="Score /100"),
        yaxis=dict(autorange="reversed"),
        height=560, margin=dict(t=50,b=0,l=0,r=60),
        plot_bgcolor="white",paper_bgcolor="white",font_color=SEWS_BLEU)
    st.plotly_chart(fig_cl, use_container_width=True)

    if not df_mh.empty:
        st.markdown("---")
        st.subheader("📈 Manhours consommées 2025")
        res = df_mh.groupby("sous_projet").agg(
            qte=("quantite","sum"),mh=("manhours","sum")
        ).reset_index()
        res = res[res["mh"]>0]
        tps_p = {r["sous_projet"]:r["tps_proto_h"]
                 for _,r in df_mh.iterrows() if pd.notna(r.get("tps_proto_h"))}
        res["mh_theo"] = res["qte"]*res["sous_projet"].map(tps_p).fillna(0)
        res["eff_pct"] = np.where(res["mh"]>0,(res["mh_theo"]/res["mh"]*100).round(1),0)
        c1,c2 = st.columns(2)
        with c1:
            fig1 = px.bar(res.sort_values("mh",ascending=False),
                          x="sous_projet",y="mh",title="Manhours réelles par projet",
                          color_discrete_sequence=[SEWS_BLEU],text_auto=".0f")
            fig1.update_layout(height=320,showlegend=False,
                               margin=dict(t=40,b=0,l=0,r=0),
                               plot_bgcolor="white",paper_bgcolor="white")
            st.plotly_chart(fig1,use_container_width=True)
        with c2:
            fig2 = px.bar(res[res["eff_pct"]>0].sort_values("eff_pct",ascending=False),
                          x="sous_projet",y="eff_pct",color="eff_pct",
                          color_continuous_scale=[SEWS_ROUGE,SEWS_ORANGE,SEWS_VERT],
                          title="Efficacité MH (%)",text_auto=".1f")
            fig2.add_hline(y=100,line_dash="dash",line_color=SEWS_BLEU)
            fig2.update_layout(height=320,coloraxis_showscale=False,
                               margin=dict(t=40,b=0,l=0,r=0),
                               plot_bgcolor="white",paper_bgcolor="white")
            st.plotly_chart(fig2,use_container_width=True)


# ═══════════════════════════════════════════════════════════
# ONGLET 3 — COUPE & ANOMALIES
# ═══════════════════════════════════════════════════════════
def onglet_coupe(tables):
    st.header("🔄 Suivi Coupe, Planning & Anomalies")
    df_coupe = tables.get("suivi_coupe",   pd.DataFrame()).copy()
    df_kx    = tables.get("ordres_komax",  pd.DataFrame()).copy()
    df_anom  = tables.get("anomalies_pmsa",pd.DataFrame()).copy()
    if df_coupe.empty:
        st.error("Données de coupe non disponibles.")
        return

    for _,r in df_coupe.iterrows():
        if r["pct_coupe"]==0:
            st.markdown(
                f'<div class="alerte-rouge">🚨 BLOQUÉ : {r["famille"]} — '
                f'{r["reference"]} — 0% coupé — Lancement : '
                f'{r["indice_lancement"]}</div>',unsafe_allow_html=True)
        elif r["pct_coupe"]<50:
            st.markdown(
                f'<div class="alerte-orange">⚠ EN RETARD : {r["famille"]} — '
                f'{r["reference"]} — {r["pct_coupe"]:.1f}%</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("✂ Situation de la coupe")
    labels   = [f"{r['famille']}\n{r['reference']}" for _,r in df_coupe.iterrows()]
    couleurs = [SEWS_VERT if p>=80 else SEWS_ORANGE if p>0 else SEWS_ROUGE
                for p in df_coupe["pct_coupe"]]
    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(name="✅ Coupé",x=labels,y=df_coupe["coupe"],
                           marker_color=couleurs,text=df_coupe["coupe"],
                           textposition="inside",textfont=dict(color="white",size=14)))
    fig_c.add_trace(go.Bar(name="⏳ Reste",x=labels,y=df_coupe["reste"],
                           marker_color=["#E8E8E8"]*len(df_coupe),
                           text=df_coupe["reste"],textposition="inside",
                           textfont=dict(color="#555",size=13)))
    for p,lbl in zip(df_coupe["pct_coupe"],labels):
        fig_c.add_annotation(x=lbl,y=df_coupe["nb_reperes"].max()*1.06,
                              text=f"<b>{p:.1f}%</b>",showarrow=False,
                              font=dict(size=13,color=SEWS_BLEU))
    fig_c.update_layout(barmode="stack",height=420,
                        margin=dict(t=50,b=30,l=0,r=0),
                        legend=dict(orientation="h",y=-0.15),
                        plot_bgcolor="white",paper_bgcolor="white",
                        font_color=SEWS_BLEU)
    st.plotly_chart(fig_c,use_container_width=True)

    df_c2 = df_coupe.copy()
    df_c2["Statut"] = df_c2["pct_coupe"].apply(
        lambda p:"🟢 OK" if p>=80 else "🟡 En cours" if p>=50
        else "🟠 En retard" if p>0 else "🔴 BLOQUÉ")
    st.dataframe(
        df_c2[["famille","reference","nb_reperes","coupe","reste","pct_coupe",
               "date_demande","date_reponse_prev","indice_lancement","Statut"
               ]].rename(columns={
            "famille":"Famille","reference":"Référence","nb_reperes":"Total",
            "coupe":"Coupé","reste":"Reste","pct_coupe":"% Coupé",
            "date_demande":"Demande","date_reponse_prev":"Réponse prévue",
            "indice_lancement":"Lancement"}),
        use_container_width=True, hide_index=True)

    if not df_kx.empty:
        st.markdown("---")
        st.subheader("⚙ Ordres KOMAX")
        c1,c2,c3 = st.columns(3)
        c1.metric("Total ordres",len(df_kx))
        c2.metric("Terminés",   int(df_kx["est_termine"].sum()),
                  f"{df_kx['est_termine'].mean()*100:.0f}%")
        c3.metric("Planifiés",  int(df_kx["est_locked"].sum()))
        stats = df_kx["Description"].value_counts().reset_index()
        stats.columns = ["Type","Nb"]
        fig_kx = px.pie(stats,names="Type",values="Nb",
                        title="Ordres par type",hole=0.4,
                        color_discrete_sequence=[SEWS_BLEU,SEWS_BLEU_LIGHT,
                                                  SEWS_VERT,SEWS_ORANGE])
        fig_kx.update_layout(height=280,margin=dict(t=40,b=0,l=0,r=0),
                              paper_bgcolor="white")
        st.plotly_chart(fig_kx,use_container_width=True)

    st.markdown("---")
    st.subheader("⏱ Chronomètre des Anomalies — IVECO Italie")
    if not df_anom.empty:
        nb_req    = (df_anom["statut"]=="Request").sum()
        nb_urgent = (df_anom["jours_attente"]>14).sum()
        c1,c2,c3 = st.columns(3)
        c1.metric("Total anomalies",len(df_anom))
        c2.metric("En attente",      nb_req)
        c3.metric("Urgentes (>14j)", nb_urgent,
                  "⚠ Action requise" if nb_urgent>0 else "OK")
        st.markdown("---")
        for i,(_,row) in enumerate(
            df_anom.sort_values("jours_attente",ascending=False).iterrows()
        ):
            jours  = int(row.get("jours_attente",30))
            draw   = str(row.get("drawing","?"))
            statut = str(row.get("statut","Request"))
            num    = int(row.get("numero",i+1))
            if jours>30:
                col,emoji,niv,bg = SEWS_ROUGE,"🔴","CRITIQUE — Action immédiate","#FFF5F5"
            elif jours>14:
                col,emoji,niv,bg = SEWS_ORANGE,"🟠","URGENT — Relance recommandée","#FFF8F0"
            elif jours>7:
                col,emoji,niv,bg = SEWS_ORANGE,"🟡","À surveiller","#FFFEF0"
            else:
                col,emoji,niv,bg = SEWS_VERT,"🟢","Dans les délais","#F0FFF4"

            cc,ci = st.columns([1,3])
            with cc:
                st.markdown(f"""
                <div style="background:{bg};border-radius:12px;padding:16px;
                            text-align:center;border:2px solid {col};
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size:11px;color:{SEWS_GRIS_DARK};">
                        Anomalie N°{num}</div>
                    <div style="font-size:48px;font-weight:900;
                                color:{col};line-height:1;">{jours}</div>
                    <div style="font-size:12px;color:{SEWS_GRIS_DARK};">
                        jours d'attente</div>
                    <div style="background:{col};color:white;border-radius:20px;
                                padding:3px 10px;font-size:11px;font-weight:bold;
                                display:inline-block;margin-top:6px;">
                        {emoji} {statut}</div>
                </div>""", unsafe_allow_html=True)
            with ci:
                st.markdown(f"""
                <div style="background:{bg};border-radius:12px;padding:16px;
                            border:2px solid {col};
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-weight:bold;color:{col};font-size:15px;
                                margin-bottom:10px;">{emoji} {niv}</div>
                    <div style="color:#333;font-size:13px;margin-bottom:6px;">
                        <b>📄 Plan :</b> {draw}</div>
                    <div style="color:#333;font-size:13px;margin-bottom:12px;">
                        <b>⏳ Attente :</b> {jours} jours / max 45</div>
                </div>""", unsafe_allow_html=True)
                st.progress(min(jours/45,1.0),
                    text=f"{jours} jours ({min(jours/45,1.0)*100:.0f}% du délai max)")
            if i<len(df_anom)-1:
                st.markdown("<hr style='margin:8px 0;opacity:0.3;'>",
                            unsafe_allow_html=True)
    else:
        st.success("✅ Aucune anomalie enregistrée")


# ═══════════════════════════════════════════════════════════
# ONGLET 4 — PRIMES
# ═══════════════════════════════════════════════════════════
def onglet_primes(tables):
    st.header("💰 Primes & Performance du mois")
    BAREME = {2:50,4:100,6:150,8:200,10:250,
              12:300,14:350,16:400,18:450,20:500}
    df_op = preparer_operateurs(tables)
    if df_op.empty:
        st.error("Données non disponibles.")
        return

    with st.expander("📋 Barème officiel SEWS Cabind"):
        b_df = pd.DataFrame([{
            "Taux":f"+{k}%",
            "Prime Prod.":f"{v//2} MAD","Prime Qual.":f"{v//2} MAD",
            "Total":f"{v} MAD","+ Bonus 0 abs.":f"{round(v*1.2)} MAD"
        } for k,v in BAREME.items()])
        st.dataframe(b_df,hide_index=True,use_container_width=True)

    st.markdown("---")
    taux_zone = {
        "BRIGLIA UREA":20,"Engine":16,"CABINA":14,
        "COFANO":8,"COFANO NDE":8,"PREMONTAGE":18,
        "CONTRÔLE ELECTRIQUE+PIN TO PIN":6,"CONTRÔLE FINAL":10,
        "SERTISSAGE +épissurage":6,"PREPARATION GAINE":8,
        "épissurage":8,"support PM":10,
        "PREPARATION ET VALIDATION LES TABLES DE MONTAGE":10,
    }
    rows = []
    for _,emp in df_op.iterrows():
        zone=str(emp.get("zone","")).strip()
        tb=taux_zone.get(zone,2)
        nb_r=int(emp["nb_jours_retard"])
        if   nb_r>=10: taux=max(2,tb-14)
        elif nb_r>=5:  taux=max(2,tb-8)
        elif nb_r>=3:  taux=max(2,tb-4)
        elif nb_r>=1:  taux=max(2,tb-2)
        else:          taux=tb
        pb=0
        for s in sorted(BAREME.keys(),reverse=True):
            if taux>=s: pb=BAREME[s]; break
        bonus=round(pb*0.20) if nb_r==0 and pb>0 else 0
        total=pb+bonus
        rows.append({
            "Opérateur":emp["nom_prenom"],"Zone":zone,"Taux":taux,
            "Taux %":f"+{taux}%","Prime Productivité":pb//2,
            "Prime Qualité":pb//2,"Bonus assiduité":bonus,
            "TOTAL MAD":total,"Nb retards":nb_r,
        })
    df_pr = pd.DataFrame(rows).sort_values("TOTAL MAD",ascending=False)
    total_p = df_pr["TOTAL MAD"].sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("💰 Total à payer",f"{total_p:,} MAD")
    c2.metric("🏆 Prime max",
              f"{df_pr['TOTAL MAD'].max()} MAD",
              df_pr.loc[df_pr['TOTAL MAD'].idxmax(),'Opérateur'].split()[0])
    c3.metric("📊 Prime moyenne",f"{df_pr['TOTAL MAD'].mean():.0f} MAD")
    c4.metric("✅ Bénéficiaires",f"{(df_pr['TOTAL MAD']>0).sum()}/{len(df_pr)}")

    st.markdown("---")
    df_a = df_pr[["Opérateur","Zone","Taux %","Prime Productivité",
                  "Prime Qualité","Bonus assiduité","TOTAL MAD","Nb retards"]].copy()
    df_a.columns = ["Opérateur","Zone","Taux","Prime Prod.","Prime Qual.",
                     "Bonus","TOTAL MAD","Retards"]
    st.dataframe(df_a,use_container_width=True,hide_index=True,height=500)

    top15 = df_pr.head(15)
    fig_p = go.Figure(go.Bar(
        x=top15["Opérateur"],y=top15["TOTAL MAD"],
        marker_color=[SEWS_VERT if v>=400 else SEWS_ORANGE if v>=200
                      else SEWS_BLEU_LIGHT for v in top15["TOTAL MAD"]],
        text=top15["TOTAL MAD"].apply(lambda v:f"{v} MAD"),
        textposition="outside"))
    fig_p.update_layout(title="Top 15 primes",yaxis_title="MAD",
                        height=380,margin=dict(t=50,b=80,l=0,r=0),
                        xaxis_tickangle=-40,plot_bgcolor="white",
                        paper_bgcolor="white",font_color=SEWS_BLEU)
    st.plotly_chart(fig_p,use_container_width=True)

    st.markdown("---")
    output = BytesIO()
    with pd.ExcelWriter(output,engine="openpyxl") as writer:
        df_a.to_excel(writer,index=False,sheet_name="Primes du mois")
    output.seek(0)
    st.download_button("⬇ Télécharger les primes (Excel)",data=output,
                       file_name="primes_sews_cabind.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    # Initialise la surveillance au premier chargement SANS relancer ETL
    initialiser_surveillance()

    # Vérifie les changements (seulement après le premier chargement)
    verifier_changements()

    afficher_entete()
    tables = charger_tout()
    afficher_sidebar(tables)

    if not os.path.exists(DB_PATH):
        st.error("Base de données introuvable. Lance : python etl_reel.py")
        return

    tab1,tab2,tab3,tab4 = st.tabs([
        "👷 Opérateurs","📊 Performance",
        "🔄 Coupe & Anomalies","💰 Primes",
    ])
    with tab1: onglet_operateurs(tables)
    with tab2: onglet_performance(tables)
    with tab3: onglet_coupe(tables)
    with tab4: onglet_primes(tables)

    st.markdown("---")
    st.caption(
        f"SEWS Cabind Maroc · MY2027 · "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')} · "
        "PFE Master Data Science ENS Martil"
    )

if __name__ == "__main__":
    main()