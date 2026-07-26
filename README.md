# Système Intelligent de Pilotage des Faisceaux Prototypes

**PFE — Master Data Science | ENS Martil | SEWS Cabind Maroc**

## Description

Système de pilotage intelligent pour le suivi des faisceaux prototypes MY2027
chez SEWS Cabind Maroc (Groupe Sumitomo Electric).

## Dashboards

- **Dashboard Encadrant** : `streamlit run app_encadrant.py`
- **Dashboard PFE** : `streamlit run app_pfe.py`

## Architecture

```
Phase 1 : Données synthétiques (generate_data_v2.py)
Phase 2 : Pipeline ETL (etl_reel.py + etl_pipeline.py)
Phase 3 : KPI dynamiques (kpi_engine.py + kpi_proto.py)
Phase 4 : Modèle ML Random Forest — F1=0.923 (ml_model_v2.py)
Phase 5 : Dashboards Streamlit (app_encadrant.py + app_pfe.py)
```

## Technologies

Python 3.11 · pandas · scikit-learn · XGBoost · SHAP ·
SQLAlchemy · SQLite · Streamlit · Plotly

## Auteur

Salma Balfoul — Encadrant : Prof. Ahmed Bendehman