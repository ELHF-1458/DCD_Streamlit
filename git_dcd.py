import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime
from io import BytesIO
import logging

# -----------------------------------------
# Configuration du logging (ici dans le fichier courant)
# -----------------------------------------
LOG_PATH = r"C:\Users\lenovo\Downloads\app_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.debug("Début de l'exécution de l'application.")

# -----------------------------------------
# Paramètres et constantes
# -----------------------------------------
# Fichier de sortie cumulant toutes les mises à jour
OUTPUT_CSV_PATH = r"C:\Users\lenovo\Downloads\Data_CD\donnees_unifiees_mis_a_jour.csv"
LOGO_PATH = os.path.join(r"C:\Users\lenovo\Downloads\Data_CD", "Centrale-Danone-Logo.png")

# Les colonnes attendues dans le fichier d'entrée sont désormais :
# "Prestataire", "Mois", "Palier kilométrique", "Année", "Cout", "Valeur"
ordre_paliers = ["[0-4000]", "[4000-8000]", "[8000-11000]", "[11011-14000]", ">14000"]
prestataires_list = ["COMPTOIR SERVICE", "S.T INDUSTRIE", "SDTM", "TRANSMEL SARL"]
couleur_barres = {2023: "#636EFA", 2024: "#EF553B", 2025: "#00B050"}

# -----------------------------------------
# Fonctions utilitaires
# -----------------------------------------
def load_data_from_uploaded(file) -> pd.DataFrame:
    """
    Lit le CSV uploadé en s’assurant que 'Mois' et 'Année' sont des entiers.
    Utilise l'encodage 'utf-8-sig' pour gérer correctement les accents.
    """
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
        if not df.empty:
            df["Mois"] = pd.to_numeric(df["Mois"], errors="coerce").astype(int)
            df["Année"] = pd.to_numeric(df["Année"], errors="coerce").astype(int)
        logging.debug("Données chargées depuis le fichier uploadé.")
        return df
    except Exception as e:
        logging.error("Erreur lors du chargement du fichier uploadé : %s", e)
        return pd.DataFrame(columns=["Prestataire", "Mois", "Palier kilométrique", "Année", "Cout", "Valeur"])

def save_data(df, csv_path):
    """Sauvegarde le DataFrame dans le CSV."""
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logging.debug("Données sauvegardées dans %s", csv_path)
    except Exception as e:
        logging.error("Erreur lors de la sauvegarde des données : %s", e)

def convert_valeur(x):
    """Convertit une valeur au format 'XX%' en float(XX)."""
    try:
        s = str(x)
        if "%" in s:
            return float(s.replace("%", ""))
        else:
            return float(s)
    except Exception as e:
        logging.error("Erreur lors de la conversion de la valeur '%s' : %s", x, e)
        return np.nan

def generate_graph(df):
    """
    Génère un line chart (courbe) à partir du DataFrame.
    Le graphique affiche l'évolution des moyennes de la "Valeur" (convertie en nombre)
    par "Palier kilométrique", pour chaque Prestataire (en facettes) et par Année (couleurs).
    La colonne "Cout" n'est pas utilisée dans le graphique.
    """
    try:
        # Conversion de "Valeur" (par exemple "43.75%") en float
        df["Valeur"] = df["Valeur"].apply(convert_valeur)
        # Transformer "Palier kilométrique" en catégorie ordonnée
        df["Palier kilométrique"] = pd.Categorical(
            df["Palier kilométrique"],
            categories=ordre_paliers,
            ordered=True
        )
        # Agréger les valeurs par (Année, Prestataire, Palier) en calculant la moyenne
        df_mean = df.groupby(["Année", "Prestataire", "Palier kilométrique"], as_index=False)["Valeur"].mean()
        df_mean.rename(columns={"Valeur": "Valeur Normalisée"}, inplace=True)
        # Générer le line chart avec Plotly Express (facettes par Prestataire)
        fig = px.line(
            df_mean,
            x="Palier kilométrique",
            y="Valeur Normalisée",
            color="Année",
            markers=True,
            facet_col="Prestataire",
            category_orders={"Palier kilométrique": ordre_paliers}
        )
        fig.update_layout(
            title=dict(
                text="<b>Évolution et Dispersion du Kilométrage par Palier et Prestataire par an</b>",
                font=dict(size=24, family="Arial", color="black")
            ),
            title_x=0.5,
            xaxis_title="Palier kilométrique",
            yaxis_title="Moyenne de la Valeur (%)",
            legend_title="Année",
            legend_title_font=dict(color="black", size=16),
            legend=dict(font=dict(color="black")),
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        # Mise à jour de la grille et des axes
        for axis in fig.layout:
            if axis.startswith("xaxis"):
                fig.layout[axis].title.font = dict(color="black", size=16)
                fig.layout[axis].tickfont = dict(color="black")
                fig.layout[axis].gridcolor = "lightgrey"
                fig.layout[axis].gridwidth = 1
            if axis.startswith("yaxis"):
                fig.layout[axis].title.font = dict(color="black", size=16)
                fig.layout[axis].tickfont = dict(color="black")
                fig.layout[axis].gridcolor = "lightgrey"
                fig.layout[axis].gridwidth = 1
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].strip(),
                                                    font=dict(size=18, color="black")))
        return fig
    except Exception as e:
        logging.error("Erreur lors de la génération du graphique : %s", e)
        st.error("Une erreur est survenue lors de la génération du graphique.")
        return None

def fig_to_png_bytes(fig):
    """Convertit la figure Plotly en PNG et renvoie un objet BytesIO."""
    try:
        img_bytes = fig.to_image(format="png", width=1900, height=900, scale=2)
        return BytesIO(img_bytes)
    except Exception as e:
        logging.error("Erreur lors de la conversion de la figure en PNG : %s", e)
        return None

# -----------------------------------------
# Configuration de l'application Streamlit
# -----------------------------------------
st.set_page_config(page_title="Dashboard Productivité - Centrale Danone", layout="wide")
st.markdown(
    """
    <style>
    .centered { display: block; margin-left: auto; margin-right: auto; }
    .title { text-align: center; font-size: 50px; font-weight: bold; }
    .subtitle { text-align: center; font-size: 20px; }
    .prestataire { text-align: center; font-size: 30px; font-weight: bold; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True
)

# Affichage du logo centré (remarquez que nous supprimons use_container_width)
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=650, output_format="PNG", caption="")
    else:
        st.write("Logo non trouvé.")

st.markdown("<h1 class='title'>Dashboard Productivité - Centrale Danone</h1>", unsafe_allow_html=True)

# -----------------------------------------
# UPLOAD du fichier d'entrée
# -----------------------------------------
uploaded_file = st.file_uploader("Uploader votre CSV d'origine (ex: donnees_unifiees_original.csv)", type=["csv"])
if uploaded_file is not None:
    df_original = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    # Convertir 'Mois' et 'Année'
    df_original["Mois"] = pd.to_numeric(df_original["Mois"], errors="coerce").fillna(0).astype(int)
    df_original["Année"] = pd.to_numeric(df_original["Année"], errors="coerce").fillna(0).astype(int)
    st.success("Fichier d'origine chargé avec succès.")
else:
    st.info("Veuillez uploader votre fichier CSV d'origine.")
    st.stop()

# Initialiser la session avec les données originales si aucune mise à jour n'existe
if "df_cum" not in st.session_state:
    st.session_state.df_cum = df_original.copy()

# -----------------------------------------
# FORMULAIRE de saisie des données (mise à jour)
# -----------------------------------------
st.subheader("Ajouter ou mettre à jour des valeurs (Cout et Valeur) pour (Année, Mois)")
annee_defaut = datetime.datetime.now().year
col_year, col_month = st.columns(2)
with col_year:
    annee = st.number_input("Année", min_value=2000, max_value=2100, value=annee_defaut, step=1)
with col_month:
    mois = st.selectbox("Mois", list(range(1, 13)))

nouvelles_lignes = []
with st.form("ajout_data"):
    for prest in prestataires_list:
        st.markdown(f"### {prest}")
        for palier in ordre_paliers:
            # Champ de saisie pour Cout
            cout_input = st.text_input(
                f"{prest} - {palier} (Cout)",
                "",
                placeholder="Entrez le coût",
                key=f"{prest}_{palier}_cout"
            ).strip()
            # Champ de saisie pour Valeur
            valeur_input = st.text_input(
                f"{prest} - {palier} (Valeur)",
                "",
                placeholder="Entrez la valeur (entre 0 et 100)",
                key=f"{prest}_{palier}_valeur"
            ).strip()
            if cout_input == "" and valeur_input == "":
                continue
            # Pour Cout, on convertit en float et formatte avec 2 décimales
            if cout_input:
                try:
                    cout_num = float(cout_input)
                    cout_str = f"{cout_num:.2f}"
                except ValueError:
                    st.warning(f"Entrée invalide pour {prest} - {palier} (Cout), ignorée.")
                    continue
            else:
                cout_str = ""
            # Pour Valeur, on convertit en float (attendu entre 0 et 100) et formatte avec "%" 
            if valeur_input:
                try:
                    valeur_num = float(valeur_input)
                    if not (0 <= valeur_num <= 100):
                        st.warning(f"Valeur hors limites pour {prest} - {palier} (Valeur), ignorée.")
                        continue
                    valeur_str = f"{valeur_num:.2f}%"
                except ValueError:
                    st.warning(f"Entrée invalide pour {prest} - {palier} (Valeur), ignorée.")
                    continue
            else:
                valeur_str = ""
            nouvelles_lignes.append({
                "Prestataire": prest,
                "Mois": mois,
                "Palier kilométrique": palier,
                "Année": annee,
                "Cout": cout_str,
                "Valeur": valeur_str
            })
    btn_submit = st.form_submit_button("Valider")
    if btn_submit and nouvelles_lignes:
        df_new = pd.DataFrame(nouvelles_lignes)
        df_new["Mois"] = df_new["Mois"].astype(int)
        df_new["Année"] = df_new["Année"].astype(int)
        df_cum = st.session_state.df_cum
        for idx, row in df_new.iterrows():
            mask = (
                (df_cum["Prestataire"] == row["Prestataire"]) &
                (df_cum["Mois"] == row["Mois"]) &
                (df_cum["Palier kilométrique"] == row["Palier kilométrique"]) &
                (df_cum["Année"] == row["Année"])
            )
            if mask.any():
                df_cum.loc[mask, "Valeur"] = row["Valeur"]
                df_cum.loc[mask, "Cout"] = row["Cout"]
            else:
                df_cum = pd.concat([df_cum, pd.DataFrame([row])], ignore_index=True)
        st.session_state.df_cum = df_cum.copy()
        st.success(f"{len(nouvelles_lignes)} mise(s) à jour effectuée(s).")

# -----------------------------------------
# Téléchargement du CSV mis à jour
# -----------------------------------------
st.subheader("Télécharger le fichier CSV mis à jour")
df_final = st.session_state.df_cum.copy()
csv_bytes = df_final.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
st.download_button("Télécharger CSV", data=csv_bytes, file_name="donnees_unifiees_mis_a_jour.csv", mime="text/csv")

# -----------------------------------------
# Génération du graphique (Line Chart)
# -----------------------------------------
st.subheader("Visualisation du Graphique")
if st.button("Générer le Graphique"):
    df_data = st.session_state.df_cum.copy()
    fig = generate_graph(df_data)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        png_bytes = fig_to_png_bytes(fig)
        if png_bytes:
            st.download_button("Télécharger le Graphique en PNG",
                               data=png_bytes,
                               file_name="graphique_productivite.png",
                               mime="image/png")
    else:
        st.error("Le graphique n'a pas pu être généré.")

logging.debug("Fin de l'exécution de l'application.")
