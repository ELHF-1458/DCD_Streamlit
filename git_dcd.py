import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
from io import BytesIO
import logging

# -----------------------------------------
# Configuration du logging
# -----------------------------------------
LOG_PATH = "app_debug.log"
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
OUTPUT_CSV_PATH = "donnees_unifiees_mis_a_jour.csv"  # Fichier de sortie
LOGO_PATH = "Centrale-Danone-Logo.png"  # Logo (doit être dans le même répertoire)
ordre_paliers = ["[0-4000]", "[4000-8000]", "[8000-11000]", "[11011-14000]", ">14000"]
# Note : On s'assure ici d'avoir la bonne orthographe sans espaces superflus.
prestataires_list = ["COMPTOIR SERVICE", "S.T INDUSTRIE", "SDTM", "TRANSMEL SARL"]
couleur_barres = {2023: "#636EFA", 2024: "#EF553B", 2025: "#00B050"}

# -----------------------------------------
# Fonctions utilitaires
# -----------------------------------------
def load_data_from_uploaded(file) -> pd.DataFrame:
    """Lit le CSV uploadé et normalise certaines colonnes."""
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
        if not df.empty:
            df["Mois"] = pd.to_numeric(df["Mois"], errors="coerce").fillna(0).astype(int)
            df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").fillna(0).astype(int)
            # Nettoyage de "Prestataire" et "Palier kilometrique"
            df["Prestataire"] = df["Prestataire"].astype(str).str.strip().str.upper()
            df["Palier kilometrique"] = df["Palier kilometrique"].astype(str).str.strip().str.upper()
        logging.debug("Données chargées et nettoyées depuis le fichier uploadé.")
        return df
    except Exception as e:
        logging.error("Erreur lors du chargement du fichier uploadé : %s", e)
        return pd.DataFrame(columns=["Prestataire", "Mois", "Palier kilometrique", "Annee", "Cout", "Valeur"])

def save_data(df, csv_path):
    """Sauvegarde le DataFrame dans un CSV."""
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
    """Convertit la valeur de 'Cout' en float en nettoyant d'éventuels caractères parasites."""
    try:
        # Transformer en chaîne, enlever les espaces (y compris insécables)
        s = str(x).replace("\xa0", "").strip()
        # Remplacer une virgule par un point, au cas où le CSV utilise la virgule pour les décimales
        s = s.replace(",", ".")
        # Si jamais il y a d'autres symboles (par exemple, un symbole monétaire), on les enlève
        s = "".join(ch for ch in s if ch.isdigit() or ch == ".")
        return float(s)
    except Exception as e:
        logging.error("Erreur lors de la conversion de Cout '%s': %s", x, e)
        return np.nan


def generate_graph(df, col_name):
    """
    Génère le graphique à partir du DataFrame cumulé sans double-normalisation
    pour la colonne indiquée ("Valeur" ou "Cout"), en masquant les barres et en
    affichant les courbes de tendance sous forme de lignes simples (ou markers si un
    seul point est présent), avec la légende des années affichée.
    """
    try:
        # Conversion de la colonne selon son type
        if col_name == "Valeur":
            df[col_name] = df[col_name].apply(convert_valeur)
        elif col_name == "Cout":
            df[col_name] = df[col_name].apply(convert_cout)
        
        # Normalisation des colonnes (supprimer espaces superflus, majuscules pour cohérence)
        df["Prestataire"] = df["Prestataire"].astype(str).str.strip().str.upper()
        df["Palier kilometrique"] = pd.Categorical(
            df["Palier kilometrique"].astype(str).str.strip().str.upper(),
            categories=ordre_paliers,
            ordered=True
        )
        
        # Agrégation des valeurs par (Annee, Prestataire, Palier kilometrique) en calculant la moyenne
        df_mean = df.groupby(["Annee", "Prestataire", "Palier kilometrique"], as_index=False)[col_name].mean()
        
        # Renommer la colonne agrégée pour la suite
        new_col = f"{col_name} Normalisé"
        df_mean.rename(columns={col_name: new_col}, inplace=True)
        
        # Création de l'histogramme (qui génère initialement des traces de type bar)
        fig = px.histogram(
            df_mean,
            x="Palier kilometrique",
            y=new_col,
            color="Annee",
            barmode="group",
            facet_col="Prestataire",
            category_orders={"Palier kilometrique": ordre_paliers},
            color_discrete_map=couleur_barres
        )
        
        # Masquer les barres de l'histogramme (nous n'utiliserons que les traces Scatter)
        fig.data = []
        
        # Mise à jour de la mise en page
        fig.update_layout(
            title=dict(
                text=f"<b>Évolution et Dispersion de {col_name} par Palier et Prestataire par an</b>",
                font=dict(size=24, family="Arial", color="black")
            ),
            title_x=0.25,
            xaxis_title="Palier kilometrique",
            yaxis_title=f"Moyenne de {col_name} (%)" if col_name == "Valeur" else f"Moyenne de {col_name}",
            legend_title="Annee",
            legend_title_font=dict(color="black", size=16),
            legend=dict(font=dict(color="black")),
            template="plotly_white",
            bargap=0.15,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        # Incliner les étiquettes de l'axe x (optionnel)
        fig.update_xaxes(tickangle=45)
        
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
        fig.for_each_annotation(lambda a: a.update(
            text=f"<b>{a.text.split('=')[-1].strip()}</b>",
            font=dict(size=18, color="black"))
        )
        
        # Ajout des courbes de tendance pour chaque groupe (Prestataire, Année)
        already_plotted = set()
        prestataires = df_mean["Prestataire"].unique()
        annees = sorted(df_mean["Annee"].unique())
        for i, prest in enumerate(prestataires):
            xaxis_name = "x" if i == 0 else f"x{i+1}"
            yaxis_name = "y" if i == 0 else f"y{i+1}"
            for annee in annees:
                df_sub = df_mean[(df_mean["Prestataire"] == prest) & (df_mean["Annee"] == annee)]
                if df_sub.empty:
                    continue
                df_sub = df_sub.sort_values("Palier kilometrique")
                # Détermine le mode à utiliser :
                # Si un seul point, on affiche uniquement le marker, sinon ligne + markers.
                mode = "markers" if df_sub.shape[0] == 1 else "lines+markers"
                
                # Affichage de la légende uniquement pour la première trace de l'année
                showlegend = (annee not in already_plotted)
                if showlegend:
                    already_plotted.add(annee)
                
                trace_trend = go.Scatter(
                    x=df_sub["Palier kilometrique"].tolist(),
                    y=df_sub[new_col].values,
                    mode=mode,
                    line=dict(
                        color=couleur_barres.get(annee, "#000000"),
                        dash="solid",         # Ligne continue
                        shape="linear",         # Ligne droite
                        width=3
                    ),
                    marker=dict(size=8),  # Taille des marqueurs, au cas où
                    name=f"Tendance {annee}",
                    legendgroup=str(annee),
                    showlegend=showlegend
                )
                trace_trend.update(xaxis=xaxis_name, yaxis=yaxis_name)
                fig.add_trace(trace_trend)
        
        return fig
    except Exception as e:
        st.error("Une erreur est survenue lors de la génération du graphique.")
        logging.error("Erreur dans generate_graph pour %s : %s", col_name, e)
        return None

def fig_to_png_bytes(fig):
    """Convertit la figure Plotly en image PNG et retourne un objet BytesIO."""
    try:
        png_bytes = fig.to_image(format="png", width=1900, height=900, scale=2)
        return BytesIO(png_bytes)
    except Exception as e:
        st.error("Erreur lors de la conversion de la figure en PNG.")
        logging.error("Erreur dans fig_to_png_bytes : %s", e)
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
    st.image(LOGO_PATH, width=650, output_format="PNG", caption="")
st.markdown("<h1 class='title'>Dashboard Productivité - Centrale Danone</h1>", unsafe_allow_html=True)

# -----------------------------------------
# Upload du fichier CSV d'origine
# -----------------------------------------
uploaded_file = st.file_uploader("Uploader votre CSV d'origine (ex: donnees_unifiees_original.csv)", type=["csv"])
if uploaded_file is not None:
    # Utilisation de la fonction de chargement qui nettoie également "Prestataire"
    df_original = load_data_from_uploaded(uploaded_file)
    st.success("Fichier d'origine chargé avec succès.")
    st.write(df_original["Cout"].shape[0])

    st.write(df_original["Valeur"].shape[0])


else:
    st.info("Veuillez uploader votre fichier CSV d'origine.")
    st.stop()

# Pour éviter toute ambiguïté, on crée initialement notre DataFrame cumulatif à partir des anciennes données
df_cum_initial = df_original.copy()

# -----------------------------------------
# FORMULAIRE de saisie pour mise à jour
# -----------------------------------------
st.subheader("Ajouter ou mettre à jour des valeurs pour (Annee, Mois)")
annee_defaut = datetime.datetime.now().year
col_year, col_month = st.columns(2)
with col_year:
    annee = st.number_input("Annee", min_value=2000, max_value=2100, value=annee_defaut, step=1)
with col_month:
    mois = st.selectbox("Mois", list(range(1, 13)))

# Vérifier si le couple (Mois, Annee) existe déjà dans les anciennes données
if ((df_cum_initial["Mois"] == mois) & (df_cum_initial["Annee"] == annee)).any():
    st.error(f"Les données pour Mois={mois} et Annee={annee} existent déjà dans le fichier d'origine.")
    st.stop()

nouvelles_lignes = []
with st.form("ajout_data"):
    for prest in prestataires_list:
        st.markdown(f"### {prest}")
        sum_valeur = 0.0
        sum_cout = 0.0
        prest_lignes = []  # Lignes pour ce prestataire
        for palier in ordre_paliers:
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
            # Vérification qu'un champ n'est pas rempli sans l'autre
            if (cout_input == "" and valeur_input != "") or (cout_input != "" and valeur_input == ""):
                st.error(f"Erreur: Pour {prest} - {palier}, remplissez à la fois 'Cout' et 'Valeur' ou laissez-les vides.")
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
                "Prestataire": prest.strip().upper(),  # On force le nettoyage ici
                "Mois": mois,
                "Palier kilometrique": palier.strip().upper(),  # On s'assure du format
                "Annee": annee,
                "Cout": f"{cout_num:.2f}",
                "Valeur": f"{valeur_num:.2f}%"  # Format attendu pour Valeur
            })
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
        # Créer un DataFrame à partir des nouvelles lignes
        df_new = pd.DataFrame(nouvelles_lignes)
        df_new["Mois"] = df_new["Mois"].astype(int)
        df_new["Annee"] = df_new["Annee"].astype(int)
        # Fusionner les anciennes données (df_cum_initial) avec les nouvelles mises à jour
        # En cas de même clés, les nouvelles valeurs remplacent les anciennes
        df_updated = df_cum_initial.copy()
        for idx, row in df_new.iterrows():
            mask = (
                (df_updated["Prestataire"] == row["Prestataire"]) &
                (df_updated["Mois"] == row["Mois"]) &
                (df_updated["Palier kilometrique"] == row["Palier kilometrique"]) &
                (df_updated["Annee"] == row["Annee"])
            )
            if mask.any():
                df_updated.loc[mask, "Valeur"] = row["Valeur"]
                df_updated.loc[mask, "Cout"] = row["Cout"]
            else:
                df_updated = pd.concat([df_updated, pd.DataFrame([row])], ignore_index=True)
        st.success(f"{len(nouvelles_lignes)} mise(s) à jour effectuée(s).")
        # Stocker le DataFrame mis à jour dans la session pour l'utiliser dans les graphiques
        st.session_state.df_updated = df_updated.copy()

# -----------------------------------------
# Téléchargement et affichage si des données mises à jour existent
# -----------------------------------------
if "df_updated" in st.session_state:
    st.subheader("Télécharger le CSV cumulé mis à jour")
    csv_bytes = st.session_state.df_updated.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
    st.download_button("Télécharger CSV", data=csv_bytes, file_name="donnees_unifiees_mis_a_jour.csv", mime="text/csv")
    
    # Affichage des graphiques cumulés pour "Valeur" et "Cout" à partir du DataFrame mis à jour
    st.subheader("Graphique pour la Valeur")
    fig_val = generate_graph(st.session_state.df_updated.copy(), "Valeur")
    if fig_val is not None:
        st.plotly_chart(fig_val, use_container_width=True)
        png_val = fig_to_png_bytes(fig_val)
        if png_val:
            st.download_button("Télécharger le graphique Valeur en PNG",
                               data=png_val, file_name="graphique_valeur.png", mime="image/png")
    
    st.subheader("Graphique pour le Cout")
    fig_cout = generate_graph(st.session_state.df_updated.copy(), "Cout")
    if fig_cout is not None:
        st.plotly_chart(fig_cout, use_container_width=True)
        png_cout = fig_to_png_bytes(fig_cout)
        if png_cout:
            st.download_button("Télécharger le graphique Cout en PNG",
                               data=png_cout, file_name="graphique_cout.png", mime="image/png")
    
logging.debug("Fin de l'exécution de l'application.")
