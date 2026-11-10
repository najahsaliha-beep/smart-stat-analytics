"""
SMART STAT ANALYTICS
Application Gradio bilingue (FR / EN) pour un pipeline statistique industriel.

IMPORTANT :
- Place le code statistique original dans un fichier nommé pipeline.py
- Ce fichier doit exposer une fonction run_pipeline(...)
- L'application tente de lui transmettre le DataFrame ou le chemin du fichier
  selon les paramètres déclarés dans run_pipeline.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Le pipeline original reste séparé pour ne pas modifier ses méthodes.
try:
    from pipeline import run_pipeline as original_run_pipeline
    PIPELINE_AVAILABLE = True
    PIPELINE_IMPORT_ERROR = ""
except Exception as exc:
    original_run_pipeline = None
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = str(exc)


APP_TITLE = "SMART STAT ANALYTICS"
APP_SUBTITLE = "Automated Statistical Pipeline | Pipeline statistique automatisé"


# ---------------------------------------------------------------------
# Traductions
# ---------------------------------------------------------------------
TEXTS = {
    "FR": {
        "tagline": "Transformez vos données en décisions intelligentes.",
        "home": "Accueil",
        "import": "Importation",
        "clean": "1. Nettoyage",
        "bayes": "2. Filtre bayésien",
        "structure": "3. Structure",
        "models": "4. Modèles statistiques",
        "dashboard": "Tableau de bord KPI",
        "results": "Résultats",
        "upload": "Importer un fichier CSV ou Excel",
        "preview": "Aperçu des données",
        "analyze": "Analyser le fichier",
        "run": "Exécuter le pipeline statistique",
        "status": "Statut",
        "kpis": "Indicateurs clés",
        "chart": "Graphique interactif",
        "x": "Variable X",
        "y": "Variable Y",
        "plot": "Générer le graphique",
        "export": "Exporter les résultats",
        "excel": "Télécharger Excel",
        "pdf": "Télécharger PDF",
        "pipeline": "État du pipeline",
        "no_file": "Veuillez importer un fichier CSV ou Excel.",
        "rows": "Nombre de lignes",
        "cols": "Nombre de colonnes",
        "missing": "Valeurs manquantes",
        "duplicates": "Lignes dupliquées",
        "numeric": "Variables numériques",
        "clean_info": "Diagnostic de qualité des données",
        "bayes_info": "Cette étape est exécutée par votre fonction originale du pipeline.",
        "structure_info": "Profil des colonnes et des types de données.",
        "models_info": "Les régressions, prévisions et le Champion Model sont exécutés par votre pipeline original.",
        "ready": "Fichier chargé avec succès.",
        "pipeline_missing": "Le fichier pipeline.py ou la fonction run_pipeline est introuvable.",
        "done": "Traitement terminé.",
        "choose_x": "Choisissez une variable X.",
        "choose_y": "Choisissez une variable Y.",
    },
    "EN": {
        "tagline": "Turn your data into intelligent decisions.",
        "home": "Home",
        "import": "Data import",
        "clean": "1. Data cleaning",
        "bayes": "2. Bayesian filter",
        "structure": "3. Data structure",
        "models": "4. Statistical models",
        "dashboard": "KPI Dashboard",
        "results": "Results",
        "upload": "Upload a CSV or Excel file",
        "preview": "Data preview",
        "analyze": "Analyze file",
        "run": "Run statistical pipeline",
        "status": "Status",
        "kpis": "Key performance indicators",
        "chart": "Interactive chart",
        "x": "X variable",
        "y": "Y variable",
        "plot": "Generate chart",
        "export": "Export results",
        "excel": "Download Excel",
        "pdf": "Download PDF",
        "pipeline": "Pipeline status",
        "no_file": "Please upload a CSV or Excel file.",
        "rows": "Number of rows",
        "cols": "Number of columns",
        "missing": "Missing values",
        "duplicates": "Duplicate rows",
        "numeric": "Numeric variables",
        "clean_info": "Data quality diagnostics",
        "bayes_info": "This step is executed by your original pipeline function.",
        "structure_info": "Column profile and data types.",
        "models_info": "Regressions, forecasts and Champion Model are executed by your original pipeline.",
        "ready": "File uploaded successfully.",
        "pipeline_missing": "pipeline.py or the run_pipeline function was not found.",
        "done": "Processing completed.",
        "choose_x": "Choose an X variable.",
        "choose_y": "Choose a Y variable.",
    },
}


def tr(language: str, key: str) -> str:
    language = language if language in TEXTS else "FR"
    return TEXTS[language].get(key, key)


# ---------------------------------------------------------------------
# Lecture des fichiers
# ---------------------------------------------------------------------
def read_dataset(file_path: str | None) -> pd.DataFrame | None:
    if not file_path:
        return None

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # Détection simple des séparateurs et encodages courants.
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    raise ValueError("Format non pris en charge. Utilisez CSV, XLSX ou XLS.")


def make_kpi_cards(df: pd.DataFrame, language: str) -> str:
    t = TEXTS[language]
    values = [
        (t["rows"], f"{len(df):,}"),
        (t["cols"], f"{len(df.columns):,}"),
        (t["missing"], f"{int(df.isna().sum().sum()):,}"),
        (t["duplicates"], f"{int(df.duplicated().sum()):,}"),
        (t["numeric"], f"{len(df.select_dtypes(include=np.number).columns):,}"),
    ]
    cards = "".join(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """
        for label, value in values
    )
    return f'<div class="kpi-grid">{cards}</div>'


def make_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame({
        "Variable": df.columns.astype(str),
        "Type": [str(dtype) for dtype in df.dtypes],
        "Valeurs manquantes": [int(df[col].isna().sum()) for col in df.columns],
        "Valeurs uniques": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })
    return profile


def load_and_preview(file_path: str | None, language: str):
    t = TEXTS[language]
    if not file_path:
        return None, None, None, t["no_file"], "", gr.update(choices=[], value=None), gr.update(choices=[], value=None)

    try:
        df = read_dataset(file_path)
        if df is None or df.empty:
            raise ValueError("Le fichier ne contient aucune donnée.")

        preview = df.head(20)
        profile = make_data_profile(df)
        numeric_cols = list(df.select_dtypes(include=np.number).columns)
        all_cols = list(df.columns)

        status = f"✅ {t['ready']}  |  {len(df)} lignes × {len(df.columns)} colonnes"
        return (
            preview,
            profile,
            make_kpi_cards(df, language),
            status,
            "",
            gr.update(choices=all_cols, value=all_cols[0] if all_cols else None),
            gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),
        )
    except Exception as exc:
        return None, None, None, f"❌ {exc}", "", gr.update(choices=[], value=None), gr.update(choices=[], value=None)


# ---------------------------------------------------------------------
# Appel du pipeline statistique original
# ---------------------------------------------------------------------
def call_original_pipeline(df: pd.DataFrame, file_path: str) -> Any:
    """
    Appelle run_pipeline sans modifier les méthodes du fichier pipeline.py.
    Les noms de paramètres les plus courants sont reconnus automatiquement.
    """
    if not PIPELINE_AVAILABLE or original_run_pipeline is None:
        raise RuntimeError(
            "pipeline.py introuvable ou import impossible. "
            f"Détail : {PIPELINE_IMPORT_ERROR}"
        )

    function = original_run_pipeline
    signature = inspect.signature(function)
    parameters = signature.parameters

    dataframe_names = {
        "df", "data", "dataframe", "dataset", "donnees", "donnees_df",
        "df_raw", "data_raw", "table"
    }
    path_names = {
        "file_path", "filepath", "path", "file", "fichier", "chemin",
        "input_path", "csv_path"
    }

    kwargs = {}
    for name, parameter in parameters.items():
        normalized = name.lower()
        if normalized in dataframe_names:
            kwargs[name] = df.copy()
        elif normalized in path_names:
            kwargs[name] = file_path

    if kwargs:
        return function(**kwargs)

    # Si la fonction possède un seul paramètre obligatoire, on lui transmet le DataFrame.
    required = [
        p for p in parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(required) == 1:
        parameter_name = required[0].name.lower()
        argument = file_path if parameter_name in path_names else df.copy()
        return function(argument)

    if len(required) == 0:
        return function()

    raise RuntimeError(
        "La signature de run_pipeline n'a pas été reconnue. "
        "Ouvre pipeline.py et indique-moi la ligne exacte de définition "
        "de run_pipeline pour adapter l'appel."
    )


def stringify_result(result: Any) -> str:
    if result is None:
        return "Le pipeline s'est exécuté. Aucun résultat n'a été retourné."
    try:
        if isinstance(result, pd.DataFrame):
            return result.to_string(index=False)
        if isinstance(result, (dict, list, tuple)):
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return str(result)
    except Exception:
        return repr(result)


def run_pipeline_ui(file_path: str | None, language: str):
    if not file_path:
        return tr(language, "no_file"), None, None

    try:
        df = read_dataset(file_path)
        result = call_original_pipeline(df, file_path)
        result_text = stringify_result(result)
        result_table = result if isinstance(result, pd.DataFrame) else None
        status = f"✅ {tr(language, 'done')}\n\n{result_text}"
        return status, result_table, result_text
    except Exception as exc:
        details = traceback.format_exc()
        return f"❌ {exc}\n\n{details}", None, ""


# ---------------------------------------------------------------------
# Graphiques interactifs
# ---------------------------------------------------------------------
def create_interactive_chart(file_path: str | None, x_col: str | None,
                             y_col: str | None, chart_type: str,
                             language: str):
    if not file_path:
        raise gr.Error(tr(language, "no_file"))

    df = read_dataset(file_path)
    if not x_col or x_col not in df.columns:
        raise gr.Error(tr(language, "choose_x"))
    if not y_col or y_col not in df.columns:
        raise gr.Error(tr(language, "choose_y"))

    data = df[[x_col, y_col]].dropna()
    if data.empty:
        raise gr.Error("Aucune donnée exploitable pour ces variables.")

    if chart_type == "Barres / Bar":
        fig = px.bar(data, x=x_col, y=y_col, title=f"{y_col} par {x_col}")
    elif chart_type == "Nuage de points / Scatter":
        fig = px.scatter(data, x=x_col, y=y_col, title=f"{y_col} vs {x_col}",
                         trendline="ols" if pd.api.types.is_numeric_dtype(data[x_col]) and
                         pd.api.types.is_numeric_dtype(data[y_col]) else None)
    elif chart_type == "Histogramme / Histogram":
        fig = px.histogram(data, x=y_col, title=f"Distribution de {y_col}")
    else:
        fig = px.line(data, x=x_col, y=y_col, title=f"Évolution de {y_col}")

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#243247"),
        title_x=0.02,
        margin=dict(l=30, r=25, t=65, b=35),
    )
    return fig


# ---------------------------------------------------------------------
# Exports Excel et PDF
# ---------------------------------------------------------------------
def export_excel(file_path: str | None, result_text: str):
    if not file_path:
        raise gr.Error("Importez d'abord un fichier.")

    df = read_dataset(file_path)
    profile = make_data_profile(df)
    output_path = Path(tempfile.gettempdir()) / (
        f"SMART_STAT_ANALYTICS_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    )

    result_df = pd.DataFrame({"Résultat du pipeline": [result_text or "Aucun résultat disponible"]})

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Donnees", index=False)
        profile.to_excel(writer, sheet_name="Profil", index=False)
        result_df.to_excel(writer, sheet_name="Resultats", index=False)

    return str(output_path)



    df = read_dataset(file_path)
    profile = make_data_profile(df)
    output_path = Path(tempfile.gettempdir()) / (
        f"SMART_STAT_ANALYTICS_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    )

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=1.7 * cm, leftMargin=1.7 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SmallBody", parent=styles["BodyText"], fontSize=8,
        leading=11, spaceAfter=5
    ))

    story = [
        Paragraph("SMART STAT ANALYTICS", styles["Title"]),
        Paragraph("Rapport d'analyse statistique", styles["Heading2"]),
        Paragraph(f"Date : {datetime.now():%d/%m/%Y %H:%M}", styles["SmallBody"]),
        Spacer(1, 10),
        Paragraph("1. Indicateurs clés", styles["Heading2"]),
    ]

    kpi_data = [
        ["Indicateur", "Valeur"],
        ["Nombre de lignes", str(len(df))],
        ["Nombre de colonnes", str(len(df.columns))],
        ["Valeurs manquantes", str(int(df.isna().sum().sum()))],
        ["Lignes dupliquées", str(int(df.duplicated().sum()))],
        ["Variables numériques", str(len(df.select_dtypes(include=np.number).columns))],
    ]
    table = Table(kpi_data, colWidths=[8 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163B63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8E1EA")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FB")]),
    ]))
    story.extend([table, Spacer(1, 12), Paragraph("2. Profil des variables", styles["Heading2"])])

    profile_data = [["Variable", "Type", "Manquantes", "Valeurs uniques"]]
    for _, row in profile.iterrows():
        profile_data.append([
            str(row["Variable"])[:28], str(row["Type"])[:18],
            str(row["Valeurs manquantes"]), str(row["Valeurs uniques"])
        ])

    profile_table = Table(profile_data, repeatRows=1,
                          colWidths=[5.4 * cm, 3.3 * cm, 3 * cm, 3.3 * cm])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163B63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E1EA")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FB")]),
    ]))
    story.extend([profile_table, Spacer(1, 12), Paragraph("3. Résultat du pipeline", styles["Heading2"])])

    safe_result = (result_text or "Le pipeline n'a pas encore été exécuté.")
    safe_result = safe_result.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for line in safe_result.splitlines()[:35]:
        if line.strip():
            story.append(Paragraph(line[:220], styles["SmallBody"]))

    doc.build(story)
    return str(output_path)


# ---------------------------------------------------------------------
# Style visuel Light
# ---------------------------------------------------------------------
CUSTOM_CSS = """
:root {
  --navy: #163B63;
  --blue: #2878B8;
  --cyan: #21A6A1;
  --ink: #243247;
  --muted: #66758A;
  --line: #E2EAF2;
  --soft: #F4F8FC;
}
.gradio-container {
  background: #F6F9FC !important;
  color: var(--ink) !important;
  max-width: 1450px !important;
}
#hero {
  background: linear-gradient(115deg, #163B63 0%, #2878B8 62%, #21A6A1 100%);
  color: white;
  padding: 30px 34px;
  border-radius: 18px;
  margin-bottom: 18px;
}
#hero h1 { color: white !important; font-size: 32px !important; margin: 0 0 8px 0; }
#hero p { color: #EAF5FF !important; margin: 0; font-size: 15px; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: 12px;
  margin: 12px 0;
}
.kpi-card {
  background: white;
  border: 1px solid #E2EAF2;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 3px 12px rgba(22,59,99,.05);
}
.kpi-label { color: #66758A; font-size: 12px; margin-bottom: 8px; }
.kpi-value { color: #163B63; font-size: 25px; font-weight: 750; }
.section-note {
  background: #EDF5FB;
  border-left: 4px solid #21A6A1;
  border-radius: 8px;
  padding: 12px 15px;
  color: #29445F;
}
button.primary {
  background: #163B63 !important;
  border-color: #163B63 !important;
}
footer { display: none !important; }
"""


# ---------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------
with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="teal",
        neutral_hue="slate"
    ),
    css=CUSTOM_CSS
) as app:

    language_state = gr.State("FR")
    result_state = gr.State("")

    gr.HTML(f"""
    <div id="hero">
      <h1>SMART STAT ANALYTICS</h1>
      <p>{APP_SUBTITLE}</p>
      <p style="margin-top:12px;font-weight:600;">Data • Statistics • Forecasting • Decision Support</p>
    </div>
    """)

    with gr.Row():
        language = gr.Radio(
            choices=[("Français / French", "FR"), ("English / Anglais", "EN")],
            value="FR", label="Langue / Language",
            scale=1
        )
        pipeline_state = gr.Markdown(
            "🟢 Interface prête | UI ready",
            scale=2
        )

    with gr.Tab("Accueil | Home"):
        gr.Markdown(
            """
            ### Bienvenue dans SMART STAT ANALYTICS
            Une application pour importer, explorer et analyser les données industrielles.

            **Workflow / Flux de travail**

            1. Importer un fichier CSV ou Excel.
            2. Examiner la qualité et la structure des données.
            3. Exécuter les quatre filtres et les méthodes statistiques du pipeline original.
            4. Visualiser les KPI et créer des graphiques interactifs.
            5. Exporter le rapport en Excel ou PDF.

            <div class="section-note">
            <b>Important :</b> les méthodes statistiques originales sont conservées dans
            <code>pipeline.py</code>. Cette interface les appelle sans les réécrire.
            </div>
            """
        )

    with gr.Tab("Importation | Data import"):
        file_input = gr.File(
            label="Importer un fichier CSV / Excel",
            file_types=[".csv", ".xlsx", ".xls"],
            type="filepath"
        )
        analyze_btn = gr.Button("Analyser le fichier | Analyze file", variant="primary")
        upload_status = gr.Markdown()
        preview_output = gr.Dataframe(label="Aperçu des données | Data preview", interactive=False)
        profile_output = gr.Dataframe(label="Profil des variables | Variable profile", interactive=False)

    with gr.Tab("1. Nettoyage | Data cleaning"):
        gr.Markdown(
            """
            ### Filtre 1 — Nettoyage des données
            Cette page présente un diagnostic des valeurs manquantes, des doublons et des types.
            Le nettoyage statistique détaillé reste effectué par la fonction originale du pipeline.
            """
        )
        cleaning_kpis = gr.HTML()
        cleaning_profile = gr.Dataframe(label="Diagnostic de qualité", interactive=False)

    with gr.Tab("2. Filtre bayésien | Bayesian filter"):
        gr.Markdown(
            """
            ### Filtre 2 — Fiabilité bayésienne
            Cette étape est exécutée par la fonction `bayesian_reliability_filter`
            de ton pipeline original lors du lancement complet.
            """
        )
        bayes_status = gr.Markdown(
            "Le filtre bayésien sera lancé avec le pipeline complet. / "
            "The Bayesian filter runs with the full pipeline."
        )

    with gr.Tab("3. Structure | Data structure"):
        gr.Markdown("### Filtre 3 — Classification et compatibilité des variables")
        structure_profile = gr.Dataframe(label="Structure des données", interactive=False)

    with gr.Tab("4. Modèles | Statistical models"):
        gr.Markdown(
            """
            ### Filtre 4 — Modèles statistiques
            Les régressions, les méthodes de prévision et la sélection du Champion Model
            sont exécutées par le code statistique original.
            """
        )
        run_btn = gr.Button("▶ Exécuter le pipeline | Run pipeline", variant="primary")
        run_status = gr.Textbox(label="Journal d'exécution | Execution log", lines=12)

    with gr.Tab("Tableau de bord KPI | KPI Dashboard"):
        gr.Markdown("### Tableau de bord — Indicateurs clés")
        dashboard_kpis = gr.HTML()
        with gr.Row():
            x_dropdown = gr.Dropdown(label="Variable X", choices=[])
            y_dropdown = gr.Dropdown(label="Variable Y", choices=[])
        chart_type = gr.Dropdown(
            label="Type de graphique | Chart type",
            choices=[
                "Ligne / Line",
                "Barres / Bar",
                "Nuage de points / Scatter",
                "Histogramme / Histogram",
            ],
            value="Ligne / Line"
        )
        chart_btn = gr.Button("Générer le graphique | Generate chart")
        chart_output = gr.Plot(label="Graphique interactif | Interactive chart")

    with gr.Tab("Résultats et export | Results & export"):
        gr.Markdown("### Résultats du pipeline")
        results_output = gr.Textbox(label="Résultat complet | Full output", lines=16)
        with gr.Row():
            excel_btn = gr.Button("Télécharger Excel | Download Excel")
            pdf_btn = gr.Button("Télécharger PDF | Download PDF")
        excel_file = gr.File(label="Fichier Excel généré")
        pdf_file = gr.File(label="Rapport PDF généré")

    # Langue sélectionnée : conserve les données, change les libellés de statut.
    def update_language(lang):
        return lang, f"🟢 Interface prête | UI ready — {lang}"

    language.change(
        update_language,
        inputs=[language],
        outputs=[language_state, pipeline_state]
    )

    # Import : met à jour l'aperçu, le profil et les listes des variables.
    def analyze_file(path, lang):
        preview, profile, cards, status, _, all_cols, numeric_cols = load_and_preview(path, lang)
        return (
            preview, profile, cards, status,
            profile, cards, all_cols, numeric_cols
        )

    analyze_btn.click(
        analyze_file,
        inputs=[file_input, language_state],
        outputs=[
            preview_output, profile_output, cleaning_profile, upload_status,
            structure_profile, dashboard_kpis, x_dropdown, y_dropdown
        ]
    )

    # Exécution du pipeline d'origine.
    def execute_pipeline(path, lang):
        status, table, result_text = run_pipeline_ui(path, lang)
        return status, result_text

    run_btn.click(
        execute_pipeline,
        inputs=[file_input, language_state],
        outputs=[run_status, results_output]
    ).then(
        lambda text: text,
        inputs=[results_output],
        outputs=[result_state]
    )

    # Graphique.
    chart_btn.click(
        create_interactive_chart,
        inputs=[file_input, x_dropdown, y_dropdown, chart_type, language_state],
        outputs=[chart_output]
    )

    # Exports.
    excel_btn.click(
        export_excel,
        inputs=[file_input, results_output],
        outputs=[excel_file]
    )
    pdf_btn.click(
        
        inputs=[file_input, results_output],
        outputs=[pdf_file]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SMART STAT ANALYTICS")
    print("=" * 60)
    if not PIPELINE_AVAILABLE:
        print("ATTENTION : pipeline.py n'est pas encore détecté.")
        print("Place ton code statistique original dans pipeline.py.")
    app.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)
