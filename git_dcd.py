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

# Colonnes attendues : "Prestataire", "Mois", "Palier kilometrique", "Annee", "Cout", "Valeur"
ordre_paliers = ["[0-4000]", "[4000-8000]", "[8000-11000]", "[11011-14000]", ">14000"]
prestataires_list = ["COMPTOIR SERVICE", "S.T INDUSTRIE", "SDTM", "TRANSMEL SARL"]
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
    Génère un line chart pour la colonne spécifiée (col_name) en agrégeant par (Annee, Prestataire, Palier kilometrique).
    Le graphique présente des courbes lissées (spline) avec des marqueurs et une épaisseur de ligne améliorée.
    """
    try:
        if col_name == "Valeur":
            df[col_name] = df[col_name].apply(convert_valeur)
        elif col_name == "Cout":
            df[col_name] = df[col_name].apply(convert_cout)
        
        df["Palier kilometrique"] = pd.Categorical(
            df["Palier kilometrique"],
            categories=ordre_paliers,
            ordered=True
        )
        df_mean = df.groupby(["Annee", "Prestataire", "Palier kilometrique"], as_index=False)[col_name].mean()
        df_mean.rename(columns={col_name: "Moyenne"}, inplace=True)
        
        # Utilisation d'une ligne spline avec marqueurs, taille des marqueurs et largeur de ligne ajustées pour une meilleure qualité
        fig = px.line(
            df_mean,
            x="Palier kilometrique",
            y="Moyenne",
            color="Annee",
            markers=True,
            facet_col="Prestataire",
            category_orders={"Palier kilometrique": ordre_paliers},
            line_shape="spline"
        )
        # Ajustement des styles
        fig.update_traces(marker=dict(size=10), line=dict(width=3))
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
    df_original["Mois"] = pd.to_numeric(df_original["Mois"], errors="coerce").fillna(0).astype(int)
    df_original["Annee"] = pd.to_numeric(df_original["Annee"], errors="coerce").fillna(0).astype(int)
    st.success("Fichier d'origine chargé avec succès.")
else:
    st.info("Veuillez uploader votre fichier CSV d'origine.")
    st.stop()

# Initialiser la base cumulée dans la session s'il n'y a pas de mises à jour
if "df_cum" not in st.session_state:
    st.session_state.df_cum = df_original.copy()

# -----------------------------------------
# FORMULAIRE de saisie (mise à jour) pour Cout et Valeur
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
        sum_valeur = 0.0
        sum_cout = 0.0
        prest_lignes = []
        for palier in ordre_paliers:
            # Les champs sont pré-remplis avec "20.00"
            cout_input = st.text_input(
                f"{prest} - {palier} (Cout)",
                value="20.00",
                key=f"{prest}_{palier}_cout"
            ).strip()
            valeur_input = st.text_input(
                f"{prest} - {palier} (Valeur)",
                value="20.00",
                key=f"{prest}_{palier}_valeur"
            ).strip()
            # Si les deux champs sont vides (ce qui ne devrait pas arriver car ils sont pré-remplis)
            if cout_input == "" and valeur_input == "":
                continue
            # Vérifier que si un champ est rempli, l'autre l'est aussi
            if (cout_input == "" and valeur_input != "") or (cout_input != "" and valeur_input == ""):
                st.error(f"Erreur: Pour {prest} - {palier}, vous devez remplir à la fois 'Cout' et 'Valeur' ou laisser les deux vides.")
                st.stop()
            try:
                cout_num = float(cout_input)
                valeur_num = float(valeur_input)
            except ValueError:
                st.error(f"Erreur de conversion pour {prest} - {palier}.")
                st.stop()
            sum_cout += cout_num
            sum_valeur += valeur_num
            prest_lignes.append({
                "Prestataire": prest,
                "Mois": mois,
                "Palier kilometrique": palier,
                "Annee": annee,
                "Cout": f"{cout_num:.2f}",
                "Valeur": f"{valeur_num:.2f}%"
            })
        # Vérifier que la somme des valeurs est exactement 100.00 pour ce prestataire
        if prest_lignes:
            if abs(sum_valeur - 100.00) > 1e-2:
                st.error(f"Erreur: La somme des 'Valeur' pour {prest} est {sum_valeur:.2f} et doit être exactement 100.00.")
                st.stop()
            if abs(sum_cout - 100.00) > 1e-2:
                st.error(f"Erreur: La somme des 'Cout' pour {prest} est {sum_cout:.2f} et doit être exactement 100.00.")
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
        st.session_state.data_updated = True

# -----------------------------------------
# Téléchargement du CSV mis à jour
# -----------------------------------------
st.subheader("Télécharger le CSV mis à jour")
df_final = st.session_state.df_cum.copy()
csv_bytes = df_final.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
st.download_button("Télécharger CSV", data=csv_bytes, file_name="donnees_unifiees_mis_a_jour.csv", mime="text/csv")

# -----------------------------------------
# Génération des graphiques (Line Charts)
# -----------------------------------------
if "data_updated" in st.session_state and st.session_state.data_updated:
    # Graphique pour Valeur
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
    # Graphique pour Cout
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
    st.session_state.data_updated = False

logging.debug("Fin de l'exécution de l'application.")
