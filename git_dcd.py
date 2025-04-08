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
# Configuration du logging
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
OUTPUT_CSV_PATH = r"C:\Users\lenovo\Downloads\Data_CD\donnees_unifiees_mis_a_jour.csv"
LOGO_PATH = os.path.join(r"C:\Users\lenovo\Downloads\Data_CD", "Centrale-Danone-Logo.png")

# Liste des catégories pour le palier et des prestataires
ordre_paliers = ["[0-4000]", "[4000-8000]", "[8000-11000]", "[11011-14000]", ">14000"]
prestataires_list = ["COMPTOIR SERVICE", "S.T INDUSTRIE", "SDTM", "TRANSMEL SARL"]
# Palette de couleurs pour les années
couleur_barres = {2023: "#636EFA", 2024: "#EF553B", 2025: "#00B050"}

# -----------------------------------------
# Fonctions utilitaires
# -----------------------------------------
def load_data_from_uploaded(file) -> pd.DataFrame:
    """Lit le CSV uploadé (encodage utf-8-sig) et convertit 'Mois' et 'Annee' en int."""
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
        if not df.empty:
            df["Mois"] = pd.to_numeric(df["Mois"], errors="coerce").fillna(0).astype(int)
            df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").fillna(0).astype(int)
        logging.debug("Données chargées depuis le fichier uploadé.")
        return df
    except Exception as e:
        logging.error("Erreur lors du chargement du fichier uploadé : %s", e)
        return pd.DataFrame(columns=["Prestataire", "Mois", "Palier kilometrique", "Annee", "Cout", "Valeur"])

def save_data(df, csv_path):
    """Sauvegarde le DataFrame dans un CSV (utf-8-sig)."""
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logging.debug("Données sauvegardées dans %s", csv_path)
    except Exception as e:
        logging.error("Erreur lors de la sauvegarde des données : %s", e)

def convert_valeur(x):
    """Convertit une chaîne du type 'XX.XX%' en float (XX.XX)."""
    try:
        s = str(x)
        if "%" in s:
            return float(s.replace("%", ""))
        else:
            return float(s)
    except Exception as e:
        logging.error("Erreur lors de la conversion de Valeur '%s' : %s", x, e)
        return np.nan

def convert_cout(x):
    """Convertit la valeur de 'Cout' en float."""
    try:
        return float(x)
    except Exception as e:
        logging.error("Erreur lors de la conversion de Cout '%s' : %s", x, e)
        return np.nan

def generate_line_chart(df, col_name):
    """
    Génère un line chart à partir du DataFrame pour la colonne spécifiée (col_name),
    en agrégeant par (Annee, Prestataire, Palier kilometrique) avec la moyenne.
    """
    try:
        # Conversion de la colonne ciblée
        if col_name == "Valeur":
            df[col_name] = df[col_name].apply(convert_valeur)
        elif col_name == "Cout":
            df[col_name] = df[col_name].apply(convert_cout)
            
        # Mettre "Palier kilometrique" en catégorie ordonnée
        df["Palier kilometrique"] = pd.Categorical(df["Palier kilometrique"],
                                                    categories=ordre_paliers,
                                                    ordered=True)
        # Agréger par (Annee, Prestataire, Palier kilometrique) en calculant la moyenne
        df_mean = df.groupby(["Annee", "Prestataire", "Palier kilometrique"], as_index=False)[col_name].mean()
        df_mean.rename(columns={col_name: "Moyenne"}, inplace=True)
        # Générer le line chart avec Plotly Express
        fig = px.line(
            df_mean,
            x="Palier kilometrique",
            y="Moyenne",
            color="Annee",
            markers=True,
            facet_col="Prestataire",
            category_orders={"Palier kilometrique": ordre_paliers}
        )
        fig.update_layout(
            title=dict(
                text=f"<b>Évolution et Dispersion du {col_name} par Palier et Prestataire par an</b>",
                font=dict(size=24, family="Arial", color="black")
            ),
            title_x=0.5,
            xaxis_title="Palier kilometrique",
            yaxis_title=f"Moyenne du {col_name}",
            legend_title="Annee",
            legend_title_font=dict(color="black", size=16),
            legend=dict(font=dict(color="black")),
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
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
        logging.error("Erreur lors de la génération du graphique pour %s : %s", col_name, e)
        st.error(f"Erreur lors de la génération du graphique pour {col_name}.")
        return None

def fig_to_png_bytes(fig):
    """Convertit la figure Plotly en PNG et retourne un BytesIO."""
    try:
        img_bytes = fig.to_image(format="png", width=1900, height=900, scale=2)
        return BytesIO(img_bytes)
    except Exception as e:
        logging.error("Erreur lors de la conversion en PNG : %s", e)
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

# Affichage du logo centré
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=650, output_format="PNG", caption="")  # Suppression de use_container_width
    else:
        st.write("Logo non trouvé.")

st.markdown("<h1 class='title'>Dashboard Productivité - Centrale Danone</h1>", unsafe_allow_html=True)

# -----------------------------------------
# UPLOAD du fichier d'entrée
# -----------------------------------------
uploaded_file = st.file_uploader("Uploader votre CSV d'origine (ex : donnees_unifiees_original.csv)", type=["csv"])
if uploaded_file is not None:
    df_original = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    df_original["Mois"] = pd.to_numeric(df_original["Mois"], errors="coerce").fillna(0).astype(int)
    df_original["Annee"] = pd.to_numeric(df_original["Annee"], errors="coerce").fillna(0).astype(int)
    st.success("Fichier d'origine chargé avec succès.")
else:
    st.info("Veuillez uploader votre fichier CSV d'origine.")
    st.stop()

# Initialiser la base cumulée dans la session (si aucune mise à jour n'est présente)
if "df_cum" not in st.session_state:
    st.session_state.df_cum = df_original.copy()

# -----------------------------------------
# FORMULAIRE de saisie (mise à jour)
# -----------------------------------------
st.subheader("Ajouter ou mettre à jour des valeurs pour (Annee, Mois)")
annee_defaut = datetime.datetime.now().year
col_year, col_month = st.columns(2)
with col_year:
    annee = st.number_input("Annee", min_value=2000, max_value=2100, value=annee_defaut, step=1)
with col_month:
    mois = st.selectbox("Mois", list(range(1, 13)))

nouvelles_lignes = []
with st.form("ajout_data"):
    for prest in prestataires_list:
        st.markdown(f"### {prest}")
        # Initialiser des sommes pour la validation
        sum_valeur = 0.0
        sum_cout = 0.0
        prest_lignes = []  # Pour stocker les lignes de ce prestataire
        for palier in ordre_paliers:
            # Champs de saisie pour Cout avec placeholder "20.00"
            cout_input = st.text_input(
                f"{prest} - {palier} (Cout)",
                "",
                placeholder="20.00",
                key=f"{prest}_{palier}_cout"
            ).strip()
            # Champs de saisie pour Valeur avec placeholder "20.00"
            valeur_input = st.text_input(
                f"{prest} - {palier} (Valeur)",
                "",
                placeholder="20.00",
                key=f"{prest}_{palier}_valeur"
            ).strip()
            if cout_input == "" and valeur_input == "":
                continue
            # Traitement de Cout
            if cout_input:
                try:
                    cout_num = float(cout_input)
                    cout_str = f"{cout_num:.2f}"
                except ValueError:
                    st.warning(f"Entrée invalide pour {prest} - {palier} (Cout), ignorée.")
                    continue
            else:
                cout_str = ""
            # Traitement de Valeur
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
            # Accumuler la somme pour validation (uniquement si une valeur est renseignée)
            if valeur_str:
                sum_valeur += valeur_num
            if cout_str:
                sum_cout += float(cout_str)
            prest_lignes.append({
                "Prestataire": prest,
                "Mois": mois,
                "Palier kilometrique": palier,
                "Annee": annee,
                "Cout": cout_str,
                "Valeur": valeur_str
            })
        # Vérifier que, pour ce prestataire, si des valeurs ont été saisies, leur somme est égale à 100.00
        if prest_lignes:
            if sum_valeur != 100.00:
                st.error(f"La somme des 'Valeur' pour {prest} est {sum_valeur:.2f}, elle doit être égale à 100.00.")
                st.stop()
            if sum_cout != 100.00:
                st.error(f"La somme des 'Cout' pour {prest} est {sum_cout:.2f}, elle doit être égale à 100.00.")
                st.stop()
            nouvelles_lignes.extend(prest_lignes)
    btn_submit = st.form_submit_button("Valider")
    if btn_submit and nouvelles_lignes:
        df_new = pd.DataFrame(nouvelles_lignes)
        df_new["Mois"] = df_new["Mois"].astype(int)
        df_new["Annee"] = df_new["Annee"].astype(int)
        df_cum = st.session_state.df_cum
        for idx, row in df_new.iterrows():
            mask = (
                (df_cum["Prestataire"] == row["Prestataire"]) &
                (df_cum["Mois"] == row["Mois"]) &
                (df_cum["Palier kilometrique"] == row["Palier kilometrique"]) &
                (df_cum["Annee"] == row["Annee"])
            )
            if mask.any():
                df_cum.loc[mask, "Valeur"] = row["Valeur"]
                df_cum.loc[mask, "Cout"] = row["Cout"]
            else:
                df_cum = pd.concat([df_cum, pd.DataFrame([row])], ignore_index=True)
        st.session_state.df_cum = df_cum.copy()
        st.success(f"{len(nouvelles_lignes)} mise(s) à jour effectuée(s).")
        # Indiquer que des données ont été mises à jour
        st.session_state.data_updated = True

# -----------------------------------------
# Téléchargement du CSV mis à jour
# -----------------------------------------
st.subheader("Télécharger le CSV mis à jour")
df_final = st.session_state.df_cum.copy()
csv_bytes = df_final.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
st.download_button("Télécharger CSV", data=csv_bytes, file_name="donnees_unifiees_mis_a_jour.csv", mime="text/csv")

# -----------------------------------------
# Génération et affichage des graphiques
# -----------------------------------------
if "data_updated" in st.session_state and st.session_state.data_updated:
    st.subheader("Graphique pour la Valeur")
    fig_val = generate_line_chart(st.session_state.df_cum.copy(), "Valeur")
    if fig_val is not None:
        st.plotly_chart(fig_val)
        png_val = fig_to_png_bytes(fig_val)
        if png_val:
            st.download_button("Télécharger le graphique Valeur en PNG",
                               data=png_val,
                               file_name="graphique_valeur.png",
                               mime="image/png")
    st.subheader("Graphique pour le Cout")
    fig_cout = generate_line_chart(st.session_state.df_cum.copy(), "Cout")
    if fig_cout is not None:
        st.plotly_chart(fig_cout)
        png_cout = fig_to_png_bytes(fig_cout)
        if png_cout:
            st.download_button("Télécharger le graphique Cout en PNG",
                               data=png_cout,
                               file_name="graphique_cout.png",
                               mime="image/png")
    # Remise à zéro du flag
    st.session_state.data_updated = False

logging.debug("Fin de l'exécution de l'application.")
