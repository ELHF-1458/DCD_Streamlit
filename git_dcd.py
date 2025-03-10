import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")


# -----------------------------------------
# Paramètres et constants
# -----------------------------------------
LOGO_PATH = "Centrale-Danone-Logo.png"  # Doit être dans le même dossier sur GitHub
ordre_paliers = ["[0-4000]", "[4000-8000]", "[8000-11000]", "[11011-14000]", ">14000"]
prestataires_list = ["COMPTOIR SERVICE", "S.T INDUSTRIE", "SDTM", "TRANSMEL SARL"]
couleur_barres = {2023: "#636EFA", 2024: "#EF553B", 2025: "#00B050"}

# -----------------------------------------
# Fonctions utilitaires
# -----------------------------------------
def convert_valeur(x):
    """Convertit 'XX%' en float(XX)."""
    s = str(x)
    if "%" in s:
        return float(s.replace("%", ""))
    else:
        return float(s)

def generate_graph(df):
    """Génère le graphique à partir du DataFrame, sans double-normalisation."""
    try:
        # 1) Convertir la colonne Valeur (type 'XX%') en float XX
        df["Valeur"] = df["Valeur"].apply(convert_valeur)
        # À présent, "Valeur" contient déjà des pourcentages

        # 2) Palier en catégorie ordonnée
        df["Palier kilométrique"] = pd.Categorical(
            df["Palier kilométrique"],
            categories=ordre_paliers,
            ordered=True
        )

        # 3) Agrégation des pourcentages (p. ex. moyenne si plusieurs lignes par (Année, Prestataire, Palier))
        #    => On NE refait plus la normalisation (x/x.sum()*100) !
        df_mean = df.groupby(["Année", "Prestataire", "Palier kilométrique"], as_index=False)["Valeur"].mean()

        # 4) Par cohérence, on renomme la colonne "Valeur" => "Valeur Normalisée" 
        #    pour le code en aval (histogramme, courbes de tendance).
        df_mean.rename(columns={"Valeur": "Valeur Normalisée"}, inplace=True)

        # 5) Histogramme Plotly
        fig = px.histogram(
            df_mean,
            x="Palier kilométrique",
            y="Valeur Normalisée",
            color="Année",
            barmode="group",
            facet_col="Prestataire",
            category_orders={"Palier kilométrique": ordre_paliers},
            color_discrete_map=couleur_barres
        )
        fig.update_layout(
            title=dict(
                text="<b>Évolution et Dispersion du Kilométrage par Palier et Prestataire par an</b>",
                font=dict(size=24, family="Arial", color="black")
            ),
            title_x=0.25,
            xaxis_title="Palier kilométrique",
            yaxis_title="Moyenne de la Valeur (%)",
            legend_title="Année",
            legend_title_font=dict(color="black", size=16),
            legend=dict(font=dict(color="black")),
            template="plotly_white",
            bargap=0.15,
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
        fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1].strip()}</b>",
                                                    font=dict(size=18, color="black")))

        # 6) Tracer les courbes de tendance lissées
        prestataires = df_mean["Prestataire"].unique()
        annees = sorted(df_mean["Année"].unique())
        for i, prest in enumerate(prestataires):
            xaxis_name = "x" if i == 0 else f"x{i+1}"
            yaxis_name = "y" if i == 0 else f"y{i+1}"
            for annee in annees:
                df_sub = df_mean[(df_mean["Prestataire"] == prest) & (df_mean["Année"] == annee)]
                if df_sub.empty or df_sub.shape[0] < 2:
                    continue
                df_sub = df_sub.sort_values("Palier kilométrique")
                trace_trend = go.Scatter(
                    x=df_sub["Palier kilométrique"].tolist(),
                    y=df_sub["Valeur Normalisée"].values,
                    mode="lines",
                    line=dict(
                        color=couleur_barres.get(annee, "#000000"),
                        dash="dash",
                        shape="spline"
                    ),
                    name=f"Tendance {annee}",
                    legendgroup=str(annee),
                    showlegend=False
                )
                trace_trend.update(xaxis=xaxis_name, yaxis=yaxis_name)
                fig.add_trace(trace_trend)

        return fig
    except Exception as e:
        st.error("Une erreur est survenue lors de la génération du graphique.")
        return None


def fig_to_png_bytes(fig):
    """Convertit la figure en PNG et renvoie l'objet BytesIO."""
    png_bytes = fig.to_image(format="png", width=1900, height=900, scale=2)
    return BytesIO(png_bytes)

# -----------------------------------------
# Config Streamlit
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

# Affichage d'un logo si présent
col1, col2, col3 = st.columns([1.5,2,1.5])
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=False, width=650)
st.markdown("<h1 class='title'>Dashboard Productivité - Centrale Danone</h1>", unsafe_allow_html=True)

# -----------------------------------------
# UPLOAD du fichier “original”
# -----------------------------------------
uploaded_file = st.file_uploader("Uploader votre CSV d'origine", type=["csv"])
if not uploaded_file:
    st.info("Veuillez uploader votre fichier d'origine pour commencer.")
    st.stop()

df_original = pd.read_csv(uploaded_file)
if df_original.empty:
    st.error("Le fichier uploadé est vide ou invalide.")
    st.stop()

# Convertir Mois/Année en int
df_original["Mois"] = pd.to_numeric(df_original["Mois"], errors="coerce").fillna(0).astype(int)
df_original["Année"] = pd.to_numeric(df_original["Année"], errors="coerce").fillna(0).astype(int)

# Stocker en session_state
if "df_cum" not in st.session_state:
    st.session_state.df_cum = df_original.copy()

# -----------------------------------------
# FORMULAIRE de saisie
# -----------------------------------------
st.subheader("Ajouter des valeurs (en %) pour un (Année, Mois)")
annee_defaut = datetime.datetime.now().year
colA, colB = st.columns(2)
with colA:
    annee = st.number_input("Année", min_value=2000, max_value=2100, value=annee_defaut)
with colB:
    mois = st.selectbox("Mois", range(1,13))

# Saisie
nouvelles_lignes = []
with st.form("ajout_data"):
    for prest in prestataires_list:
        st.markdown(f"### {prest}")
        for palier in ordre_paliers:
            val_str = st.text_input(
                f"{prest} - {palier}",
                "",
                placeholder="Entre 0 et 100",
                key=f"{prest}_{palier}"
            ).strip()
            if val_str:
                try:
                    val_float = float(val_str)
                    if 0<=val_float<=100:
                        val_fmt = f"{val_float:.2f}%"
                        nouvelles_lignes.append({
                            "Prestataire": prest,
                            "Mois": mois,
                            "Palier kilométrique": palier,
                            "Année": annee,
                            "Valeur": val_fmt
                        })
                    else:
                        st.warning(f"Valeur hors limite pour {prest} - {palier}, ignorée.")
                except:
                    st.warning(f"Entrée invalide pour {prest} - {palier}, ignorée.")
    btn_submit = st.form_submit_button("Valider")

if btn_submit and nouvelles_lignes:
    df_new = pd.DataFrame(nouvelles_lignes)
    df_cum = st.session_state.df_cum
    # Mettre à jour / ajouter
    for idx, row in df_new.iterrows():
        mask = (
            (df_cum["Prestataire"] == row["Prestataire"]) &
            (df_cum["Mois"] == row["Mois"]) &
            (df_cum["Palier kilométrique"] == row["Palier kilométrique"]) &
            (df_cum["Année"] == row["Année"])
        )
        if mask.any():
            df_cum.loc[mask,"Valeur"] = row["Valeur"]
        else:
            df_cum = pd.concat([df_cum, pd.DataFrame([row])], ignore_index=True)
    st.session_state.df_cum = df_cum.copy()
    st.success(f"{len(nouvelles_lignes)} mise(s) à jour effectuée(s).")

# -----------------------------------------
# Bouton pour télécharger le CSV mis à jour
# -----------------------------------------
st.subheader("Télécharger le CSV mis à jour")
df_final = st.session_state.df_cum.copy()
csv_bytes = df_final.to_csv(index=False).encode("utf-8")
st.download_button("Télécharger CSV", data=csv_bytes, file_name="donnees_unifiees_mis_a_jour.csv")

# -----------------------------------------
# Bouton pour GÉNÉRER LE GRAPHIQUE
# -----------------------------------------
st.subheader("Visualisation du Graphique")
if st.button("Générer le Graphique"):
    fig = generate_graph(df_final)
    st.plotly_chart(fig, use_container_width=True)
    png_bytes = fig_to_png_bytes(fig)
    if png_bytes:
        st.download_button("Télécharger le Graphique en PNG",
                           data=png_bytes,
                           file_name="graphique_productivite.png",
                           mime="image/png")
