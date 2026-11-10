
import gradio as gr
import pandas as pd
import numpy as np
import json
import tempfile
import os
import pipeline as pl


# =========================================================
# SMART STAT ANALYTICS - SALIHA NAJAH
# =========================================================

APP_NAME = "SMART STAT ANALYTICS"
AUTHOR = "SALIHA NAJAH"


# =========================================================
# 1. THEME BLUE + TURQUOISE
# =========================================================

custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate
).set(
    body_background_fill="#F2F6FA",
    body_text_color="#10243A",
    block_background_fill="#FFFFFF",
    block_border_color="#DCE5ED",
    button_primary_background_fill="#176B87",
    button_primary_background_fill_hover="#10243A",
    button_primary_text_color="#FFFFFF"
)

custom_css = """
.gradio-container {
    max-width: 1350px !important;
    margin: auto !important;
    background: #F2F6FA !important;
}

h1 {
    color: #10243A !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #176B87 !important;
}

button.primary {
    background: #176B87 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 9px !important;
}

button.primary:hover {
    background: #10243A !important;
}

.block {
    border-radius: 12px !important;
    border-color: #DCE5ED !important;
}

footer {
    display: none !important;
}
"""


# =========================================================
# 2. IMPORTATION DES DONNEES
# =========================================================

def charger_fichier(file):

    if file is None:
        raise gr.Error("Veuillez sélectionner un fichier.")

    try:
        df = pl.load_dataset(file)

        if df.empty:
            raise gr.Error("Le fichier est vide.")

        numeric = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        dropdown = gr.update(
            choices=numeric,
            value=numeric[0] if numeric else None
        )

        message = (
            f"Fichier chargé avec succès : "
            f"{len(df)} lignes × {len(df.columns)} colonnes."
        )

        return df, message, dropdown, df, df

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(f"Erreur de chargement : {str(e)}")


# =========================================================
# 3. FILTRAGE DES DONNEES
# =========================================================

def appliquer_filtres(df, colonne, minimum, maximum):

    if df is None:
        raise gr.Error("Veuillez importer les données.")

    resultat = df.copy()

    if colonne:

        valeurs = pd.to_numeric(
            resultat[colonne],
            errors="coerce"
        )

        if minimum is not None:
            resultat = resultat[valeurs >= minimum]

        valeurs = pd.to_numeric(
            resultat[colonne],
            errors="coerce"
        )

        if maximum is not None:
            resultat = resultat[valeurs <= maximum]

    return (
        resultat,
        f"{len(resultat)} lignes conservées après filtrage.",
        resultat
    )


def reinitialiser_filtres(df):

    if df is None:
        raise gr.Error("Veuillez importer un fichier.")

    return (
        df,
        "Filtres réinitialisés.",
        df
    )


# =========================================================
# 4. PIPELINE STATISTIQUE COMPLET
# =========================================================

def analyser_pipeline(df):

    if df is None or df.empty:
        raise gr.Error("Veuillez importer des données.")

    try:

        # NIVEAU 1 : DATA CLEANING

        clean, cleaning_log = pl.clean_dataset(df)

        if clean.empty:
            raise gr.Error(
                "Aucune donnée disponible après nettoyage."
            )

        # NIVEAU 2 : BAYESIAN FILTER

        clean, bayes_report = pl.bayesian_reliability_filter(
            clean,
            cleaning_log
        )

        # NIVEAU 3 : STRUCTURE ET COMPATIBILITE

        profile = pl.profile_structure(clean)

        roles = pl.validate_roles(
            clean,
            profile
        )

        temporal = pl.temporal_compatibility(
            clean,
            roles
        )

        causal = pl.causal_compatibility(
            clean,
            roles
        )

        classification = pl.final_classification(
            temporal,
            causal
        )

        # NIVEAU 4 : MODELISATION

        causal_results = None
        ts_results = None

        if causal.get("status") == "POTENTIAL":

            causal_results = pl.run_causal_regressions(
                clean,
                roles
            )

        if temporal.get("status") == "YES":

            ts_results = pl.evaluate_time_series_candidates(
                temporal["series"],
                period=temporal.get(
                    "period_used_for_stl"
                ),
                trend_present=temporal.get(
                    "trend_present", False
                ),
                seasonality_present=temporal.get(
                    "seasonality_present", False
                )
            )

        # COMPARAISON GLOBALE DES MODELES

        scores = {}
        predictions = {}
        actuals = {}

        for family, result in [
            ("Causal", causal_results),
            ("Temporel", ts_results)
        ]:

            if result and result.get("status") == "OK":

                for model, metrics in result["scores"].items():

                    name = f"{family} - {model}"

                    scores[name] = metrics

                    predictions[name] = (
                        result["predictions"][model]
                    )

                    actuals[name] = (
                        result["test_actuals"][model]
                    )

        # SELECTION DU MEILLEUR MODELE

        if scores:

            scores_df = pd.DataFrame(scores).T

            scores_df.index.name = "Famille et modèle"

            scores_df = (
                scores_df
                .reset_index()
                .sort_values("RMSE")
            )

            best_model = scores_df.iloc[0][
                "Famille et modèle"
            ]

            best_metrics = scores[best_model]

            best_df = pd.DataFrame({
                "Valeur réelle": actuals[best_model],
                "Prévision": predictions[best_model]
            })

            mape = best_metrics["MAPE_percent"]

            best_text = (
                f"MODELE GAGNANT : {best_model}\n\n"
                f"MAE : {best_metrics['MAE']:.4f}\n"
                f"RMSE : {best_metrics['RMSE']:.4f}\n"
                f"MAPE : "
                f"{f'{mape:.2f} %' if mape is not None else 'Non disponible'}"
            )

        else:

            scores_df = pd.DataFrame()
            best_df = pd.DataFrame()

            best_text = (
                "Aucun modèle n'a pu être évalué "
                "avec les données disponibles."
            )

        # RAPPORTS JSON

        temporal_report = {
            k: v for k, v in temporal.items()
            if k != "series"
        }

        cleaning_json = json.dumps(
            pl.json_safe(cleaning_log),
            ensure_ascii=False,
            indent=2,
            default=str
        )

        bayes_json = json.dumps(
            pl.json_safe(bayes_report),
            ensure_ascii=False,
            indent=2,
            default=str
        )

        diagnostic = {
            "Structure": profile,
            "Roles": roles,
            "Compatibilite temporelle": temporal_report,
            "Compatibilite causale": causal,
            "Classification finale": classification
        }

        diagnostic_json = json.dumps(
            pl.json_safe(diagnostic),
            ensure_ascii=False,
            indent=2,
            default=str
        )

        # SAUVEGARDE DES RESULTATS

        state = {
            "clean": clean,
            "cleaning_log": cleaning_log,
            "bayes": bayes_report,
            "diagnostic": diagnostic,
            "scores": scores_df,
            "best": best_df
        }

        return (
            clean,
            cleaning_json,
            bayes_json,
            diagnostic_json,
            scores_df,
            best_text,
            best_df,
            state
        )

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(
            f"Erreur pendant l'analyse : {str(e)}"
        )


# =========================================================
# 5. EXPORTATION EXCEL
# =========================================================

def exporter_resultats(state):

    if not state:
        raise gr.Error(
            "Veuillez lancer le pipeline avant l'exportation."
        )

    try:

        fd, chemin = tempfile.mkstemp(
            suffix=".xlsx"
        )

        os.close(fd)

        with pd.ExcelWriter(
            chemin,
            engine="openpyxl"
        ) as writer:

            state["clean"].to_excel(
                writer,
                sheet_name="Donnees_nettoyees",
                index=False
            )

            state["scores"].to_excel(
                writer,
                sheet_name="Comparaison_modeles",
                index=False
            )

            state["best"].to_excel(
                writer,
                sheet_name="Modele_gagnant",
                index=False
            )

            # Conversion des valeurs complexes en texte
            cleaning_data = [
                {
                    "Indicateur": key,
                    "Valeur": json.dumps(
                        pl.json_safe(value),
                        ensure_ascii=False,
                        default=str
                    )
                }
                for key, value in state["cleaning_log"].items()
            ]

            pd.DataFrame(cleaning_data).to_excel(
                writer,
                sheet_name="Nettoyage",
                index=False
            )

            bayes_data = [
                {
                    "Indicateur": key,
                    "Valeur": json.dumps(
                        pl.json_safe(value),
                        ensure_ascii=False,
                        default=str
                    )
                }
                for key, value in state["bayes"].items()
            ]

            pd.DataFrame(bayes_data).to_excel(
                writer,
                sheet_name="Fiabilite_bayesienne",
                index=False
            )

            diagnostic_data = pd.DataFrame([
                {
                    "Diagnostic": json.dumps(
                        pl.json_safe(state["diagnostic"]),
                        ensure_ascii=False,
                        default=str
                    )
                }
            ])

            diagnostic_data.to_excel(
                writer,
                sheet_name="Diagnostic",
                index=False
            )

        return chemin

    except Exception as e:
        raise gr.Error(
            f"Erreur exportation : {str(e)}"
        )


# =========================================================
# 6. INTERFACE GRADIO
# =========================================================

with gr.Blocks(
    theme=custom_theme,
    css=custom_css,
    title=APP_NAME
) as demo:

    raw_state = gr.State(None)
    filtered_state = gr.State(None)
    analysis_state = gr.State(None)

    # -----------------------------------------------------
    # EN-TETE PRINCIPAL
    # -----------------------------------------------------

    gr.Markdown("""
    <div style="
        background: linear-gradient(135deg, #10243A, #176B87);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 15px;
    ">
        <h1 style="color: white !important;
                   font-size: 34px;">
            SMART STAT ANALYTICS
        </h1>

        <h3 style="color: #35D0C0 !important;">
            Statistical Intelligence Platform
        </h3>

        <p>Automated Statistical Pipeline</p>

        <hr style="border-color: #35D0C0;">

        <h3 style="color: white !important;">
            Developed by SALIHA NAJAH
        </h3>

        <p>Industrial Engineering Student</p>

        <p>Operational Excellence & Smart Manufacturing</p>
    </div>
    """)

    # =====================================================
    # ONGLET 1 : ACCUEIL
    # =====================================================

    with gr.Tab("Accueil"):

        gr.Markdown("""
        ## Bienvenue dans SMART STAT ANALYTICS

        Une plateforme intelligente destinée à automatiser
        l'analyse statistique des données industrielles.

        ### Nos fonctionnalités

        - Importation des fichiers Excel et CSV
        - Nettoyage automatique des données
        - Filtrage des variables
        - Fiabilité bayésienne
        - Diagnostic temporel et causal
        - Modélisation statistique
        - Comparaison des modèles
        - Sélection du modèle gagnant
        - Exportation des résultats en Excel

        ---

        ### Objectif

        Transformer les données en informations utiles
        pour améliorer la prise de décision.
        """)

    # =====================================================
    # ONGLET 2 : IMPORTATION
    # =====================================================

    with gr.Tab("Importation"):

        gr.Markdown("## Importation des données")

        fichier = gr.File(
            label="Choisir un fichier Excel ou CSV",
            file_types=[".xlsx", ".xls", ".csv"],
            type="filepath"
        )

        charger = gr.Button(
            "CHARGER LES DONNEES",
            variant="primary"
        )

        message_import = gr.Textbox(
            label="Statut de chargement"
        )

        apercu = gr.Dataframe(
            label="Aperçu des données",
            interactive=False
        )

    # =====================================================
    # ONGLET 3 : FILTRES
    # =====================================================

    with gr.Tab("Filtres"):

        gr.Markdown("## Filtrage des données")

        colonne = gr.Dropdown(
            label="Choisir une variable numérique",
            choices=[]
        )

        minimum = gr.Number(
            label="Valeur minimale",
            precision=None
        )

        maximum = gr.Number(
            label="Valeur maximale",
            precision=None
        )

        with gr.Row():

            bouton_filtrer = gr.Button(
                "APPLIQUER LES FILTRES",
                variant="primary"
            )

            bouton_reset = gr.Button(
                "REINITIALISER"
            )

        message_filtre = gr.Textbox(
            label="Statut du filtrage"
        )

        tableau_filtre = gr.Dataframe(
            label="Données filtrées",
            interactive=False
        )

    # =====================================================
    # ONGLET 4 : DATA CLEANING
    # =====================================================

    with gr.Tab("Data Cleaning"):

        gr.Markdown("""
        ## NIVEAU 1 : DATA CLEANING

        - Traitement des valeurs manquantes
        - Suppression des doublons
        - Détection des valeurs aberrantes
        - Imputation des valeurs manquantes
        """)

        tableau_clean = gr.Dataframe(
            label="Données nettoyées",
            interactive=False
        )

        rapport_clean = gr.Code(
            label="Rapport de nettoyage",
            language="json"
        )

    # =====================================================
    # ONGLET 5 : BAYESIAN FILTER
    # =====================================================

    with gr.Tab("Bayesian Filter"):

        gr.Markdown("""
        ## NIVEAU 2 : BAYESIAN RELIABILITY FILTER

        Méthode Beta-Binomial.

        Calcul de la fiabilité estimée et de
        l'intervalle crédible à 95 %.
        """)

        rapport_bayes = gr.Code(
            label="Résultats bayésiens",
            language="json"
        )

    # =====================================================
    # ONGLET 6 : DIAGNOSTIC
    # =====================================================

    with gr.Tab("Diagnostic"):

        gr.Markdown("""
        ## NIVEAU 3 : STRUCTURE ET COMPATIBILITE

        - Identification des variables
        - Détection des variables temporelles
        - Compatibilité causale
        - Classification finale
        """)

        diagnostic_output = gr.Code(
            label="Rapport de diagnostic",
            language="json"
        )

    # =====================================================
    # ONGLET 7 : MODELISATION
    # =====================================================

    with gr.Tab("Modélisation"):

        gr.Markdown("""
        ## NIVEAU 4 : MODELISATION ET SELECTION

        Régressions causales et modèles de prévision
        des séries temporelles.

        **Métriques : MAE, RMSE et MAPE.**
        """)

        bouton_analyse = gr.Button(
            "LANCER LE PIPELINE COMPLET",
            variant="primary"
        )

        scores_output = gr.Dataframe(
            label="Comparaison des modèles",
            interactive=False
        )

        gagnant_output = gr.Textbox(
            label="Modèle gagnant",
            lines=5
        )

        predictions_output = gr.Dataframe(
            label="Valeurs réelles et prévisions",
            interactive=False
        )

    # =====================================================
    # ONGLET 8 : EXPORTATION
    # =====================================================

    with gr.Tab("Exportation"):

        gr.Markdown("""
        ## Exportation des résultats

        Générer un rapport Excel comprenant :

        - Données nettoyées
        - Comparaison des modèles
        - Modèle gagnant
        - Rapport de nettoyage
        - Fiabilité bayésienne
        - Diagnostic statistique
        """)

        bouton_export = gr.Button(
            "GENERER LE RAPPORT EXCEL",
            variant="primary"
        )

        fichier_export = gr.File(
            label="Télécharger le rapport"
        )

    # =====================================================
    # ONGLET 9 : A PROPOS
    # =====================================================

    with gr.Tab("À propos"):

        gr.Markdown("""
        # SMART STAT ANALYTICS

        ### Statistical Intelligence Platform

        **Interface développée par : SALIHA NAJAH**

        Étudiante ingénieure en Génie Industriel.

        Option : Excellence Opérationnelle & Smart Manufacturing.

        **Pipeline statistique automatisé par : AMZIL Chaymaa**

        ### Technologies utilisées

        - Python
        - Pandas
        - NumPy
        - SciPy
        - Scikit-learn
        - Statsmodels
        - Gradio

        ---

        Statistical Data Analysis & Smart Manufacturing.
        """)

    # =====================================================
    # CONNEXION DES BOUTONS
    # =====================================================

    # Importation
    charger.click(
        charger_fichier,
        inputs=fichier,
        outputs=[
            raw_state,
            message_import,
            colonne,
            filtered_state,
            apercu
        ]
    )

    # Application des filtres
    bouton_filtrer.click(
        appliquer_filtres,
        inputs=[
            raw_state,
            colonne,
            minimum,
            maximum
        ],
        outputs=[
            filtered_state,
            message_filtre,
            tableau_filtre
        ]
    )

    # Réinitialisation
    bouton_reset.click(
        reinitialiser_filtres,
        inputs=raw_state,
        outputs=[
            filtered_state,
            message_filtre,
            tableau_filtre
        ]
    )

    # Exécution du pipeline
    bouton_analyse.click(
        analyser_pipeline,
        inputs=filtered_state,
        outputs=[
            tableau_clean,
            rapport_clean,
            rapport_bayes,
            diagnostic_output,
            scores_output,
            gagnant_output,
            predictions_output,
            analysis_state
        ]
    )

    # Exportation Excel
    bouton_export.click(
        exporter_resultats,
        inputs=analysis_state,
        outputs=fichier_export
    )


# =========================================================
# 7. LANCEMENT DE L'APPLICATION
# =========================================================

if __name__ == "__main__":
 demo.launch(share=True) 