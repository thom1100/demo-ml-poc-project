from __future__ import annotations

import pandas as pd
import streamlit as st

from pathlib import Path
from config import MODEL_METRICS_FILE, PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
DATA_DIR = PROJECT_ROOT / "data"

def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV file if it exists, otherwise return None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    """Read a parquet file if it exists, otherwise return None."""
    if path.exists():
        return pd.read_parquet(path)
    return None


def _display_project_intro() -> None:
    """Render the project introduction."""
    st.title("Machine Learning Proof of Concept")
    st.markdown(
        """
        Cette application présente les principales étapes du projet :
        - compréhension du problème
        - exploration des données
        - feature engineering
        - entraînement et comparaison de plusieurs modèles
        - visualisation des résultats
        """
    )

    st.subheader("Objectif du projet")
    st.write(
        """
        L'objectif est de prédire le volume de vélos comptés à partir de variables
        temporelles et éventuellement d'autres variables explicatives. Le projet
        suit une démarche complète de machine learning, de la donnée brute jusqu'à
        la comparaison des modèles dans une application Streamlit.
        """
    )


def _display_dataset_overview() -> None:
    """Render a quick dataset overview if training data is available."""
    st.subheader("Aperçu des données")

    train_path = DATA_DIR / "raw/bikes.parquet"
    df = _safe_read_parquet(train_path)

    if df is None:
        st.info(
            "Le fichier `data/train.parquet` n'est pas disponible. "
            "Ajoutez les données pour afficher l'aperçu et l'EDA."
        )
        return

    st.write(f"Nombre de lignes : **{df.shape[0]}**")
    st.write(f"Nombre de colonnes : **{df.shape[1]}**")

    with st.expander("Voir un échantillon des données"):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Voir les types de variables"):
        dtype_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(dtype) for dtype in df.dtypes],
            }
        )
        st.dataframe(dtype_df, use_container_width=True)

    with st.expander("Statistiques descriptives"):
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            st.info("Aucune variable numérique disponible.")
        else:
            st.dataframe(numeric_df.describe().T, use_container_width=True)


def _display_metrics() -> None:
    """Render model evaluation metrics."""
    st.subheader("Comparaison des modèles")

    if not MODEL_METRICS_FILE.exists():
        st.info(
            "Le fichier `results/model_metrics.csv` n'existe pas encore. "
            "Lancez `python scripts/main.py` après avoir entraîné et sauvegardé vos modèles."
        )
        return

    metrics_df = pd.read_csv(MODEL_METRICS_FILE)

    if metrics_df.empty:
        st.warning("Le fichier de métriques est vide.")
        return

    st.dataframe(metrics_df, use_container_width=True)

    numeric_columns = [
        col
        for col in metrics_df.columns
        if col not in {"model_key", "model_name", "model_path"}
        and pd.api.types.is_numeric_dtype(metrics_df[col])
    ]

    if numeric_columns:
        selected_metric = st.selectbox(
            "Choisir une métrique à visualiser",
            options=numeric_columns,
            index=0,
        )

        chart_df = metrics_df.copy()

        if "model_name" in chart_df.columns:
            chart_df = chart_df.set_index("model_name")
        else:
            chart_df = chart_df.set_index("model_key")

        st.bar_chart(chart_df[selected_metric])


def _display_saved_plots() -> None:
    """Render plots saved during notebook experimentation or training."""
    st.subheader("Visualisations enregistrées")

    if not PLOTS_DIR.exists():
        st.info(
            "Le dossier `results/plots/` n'existe pas encore. "
            "Générez des figures depuis le notebook ou vos scripts."
        )
        return

    plot_files = sorted(
        [
            path
            for path in PLOTS_DIR.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )

    if not plot_files:
        st.info("Aucune image trouvée dans `results/plots/`.")
        return

    selected_plot = st.selectbox(
        "Choisir une visualisation",
        options=plot_files,
        format_func=lambda path: path.name,
    )

    st.image(str(selected_plot), caption=selected_plot.name, use_container_width=True)

    with st.expander("Afficher toutes les visualisations"):
        for plot_file in plot_files:
            st.image(str(plot_file), caption=plot_file.name, use_container_width=True)


def _display_methodology() -> None:
    """Render a pedagogical summary of the project workflow."""
    st.subheader("Méthodologie du projet")
    st.markdown(
        """
        ### Étapes suivies
        1. **Création du repository GitHub**
           - structuration du projet
           - versionnement du code
        2. **Choix du sujet de machine learning**
           - définition du problème
           - choix de la variable cible
        3. **Recherche et préparation des données**
           - collecte
           - nettoyage
           - traitement des valeurs manquantes
        4. **Feature engineering**
           - extraction de variables temporelles
           - transformation de colonnes utiles au modèle
        5. **Entraînement de plusieurs modèles**
           - baseline
           - modèles plus avancés
        6. **Comparaison des performances**
           - métriques
           - interprétation des résultats
        7. **Communication**
           - application Streamlit
           - visualisations
           - conclusion
        """
    )


def _display_conclusion() -> None:
    """Render the project conclusion section."""
    st.subheader("Conclusion")
    st.write(
        """
        Cette application sert de vitrine au projet. Elle permet de présenter
        de manière claire :
        - les données utilisées
        - les transformations réalisées
        - les modèles testés
        - les résultats obtenus
        """
    )


def build_app() -> None:
    """Render the project Streamlit application."""
    st.set_page_config(page_title="ML Project Template", layout="wide")

    _display_project_intro()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Présentation",
            "Données",
            "Modèles",
            "Visualisations",
            "Méthodologie",
        ]
    )

    with tab1:
        _display_project_intro()
        _display_conclusion()

    with tab2:
        _display_dataset_overview()

    with tab3:
        _display_metrics()

    with tab4:
        _display_saved_plots()

    with tab5:
        _display_methodology()

def build_app() -> None:
    """Render the project Streamlit application.

    Students should replace the placeholder sections with their own visualizations,
    explanations, and prediction workflow. The function name and file location are
    fixed because ``scripts/main.py`` launches Streamlit with this module.
    """

    st.set_page_config(page_title="ML Project Template", layout="wide")

    _display_project_intro()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Présentation",
            "Données",
            "Modèles",
            "Visualisations",
            "Méthodologie",
        ]
    )

    with tab1:
        _display_project_intro()
        _display_conclusion()

    with tab2:
        _display_dataset_overview()

    with tab3:
        _display_metrics()

    with tab4:
        _display_saved_plots()

    with tab5:
        _display_methodology()


if __name__ == "__main__":
    build_app()
