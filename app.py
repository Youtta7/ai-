import os
import joblib
import pandas as pd
import streamlit as st
from src.preprocessing import tokeniser

st.set_page_config(
    page_title="Analyse Sentiment Réclamations",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Analyse Automatique du Sentiment des Réclamations")
st.write(
    "Testez une ou plusieurs réclamations, importez un fichier CSV, et"
    " consultez les tendances globales."
)

# 1. Chargement du modèle entraîné
MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
DATA_PATH = os.path.join("data", "reclamations.csv")


@st.cache_resource
def charger_modele(chemin_modele: str):
    return joblib.load(chemin_modele)


try:
    model = charger_modele(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Modèle introuvable (`models/sentiment_model.pkl`). Lancez d'abord"
        " `python src/model.py` dans le terminal."
    )
    st.stop()


def predire_liste(textes: list[str]) -> pd.DataFrame:
    preds = model.predict(textes)
    if hasattr(model, "predict_proba"):
        probs_list = model.predict_proba(textes)
        confiances = [round(float(max(p)) * 100, 1) for p in probs_list]
        return pd.DataFrame({
            "Réclamation": textes,
            "Sentiment": preds,
            "Confiance (%)": confiances,
        })
    return pd.DataFrame({
        "Réclamation": textes,
        "Sentiment": preds,
    })


def afficher_stats_sentiment(df_sent: pd.DataFrame, colonne_sent: str):
    total = len(df_sent)
    if total == 0:
        st.info("Aucune donnée à afficher.")
        return

    counts = df_sent[colonne_sent].value_counts()

    pos_count = int(counts.get("Positif", counts.get("positif", 0)))
    neu_count = int(counts.get("Neutre", counts.get("neutre", 0)))
    neg_count = int(
        counts.get("Négatif", counts.get("Negatif", counts.get("negatif", 0)))
    )

    pos_pct = (pos_count / total) * 100
    neu_pct = (neu_count / total) * 100
    neg_pct = (neg_count / total) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Réclamations", total)
    col2.metric("Positif", f"{pos_count}", f"{pos_pct:.1f}%")
    col3.metric("Neutre", f"{neu_count}", f"{neu_pct:.1f}%")
    col4.metric("Négatif", f"{neg_count}", f"{neg_pct:.1f}%")

    df_chart = pd.DataFrame(
        {
            "Sentiment": ["Positif", "Neutre", "Négatif"],
            "Nombre": [pos_count, neu_count, neg_count],
        }
    ).set_index("Sentiment")

    st.bar_chart(df_chart, height=300)


# 2. Analyse d'une ou plusieurs réclamations (texte libre)
st.subheader("🔍 Analyser une ou plusieurs réclamations")
texte = st.text_area(
    "Entrez une ou plusieurs réclamations client (une par ligne, en français)",
    height=130,
    placeholder=(
        "Exemple :\n"
        "Le colis est arrivé en retard et le service client ne répond pas.\n"
        "Livraison rapide et produit conforme, merci !"
    ),
)

if st.button("Analyser le sentiment", type="primary"):
    lignes = [ligne.strip() for ligne in texte.splitlines() if ligne.strip()]

    if not lignes:
        st.warning("Veuillez saisir une réclamation avant de lancer l'analyse.")
    elif len(lignes) == 1:
        pred = model.predict([lignes[0]])[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([lignes[0]])[0]
            confiance = float(max(probs)) * 100
            st.success(
                f"**Sentiment détecté :** {pred} — **Confiance :** {confiance:.1f}%"
            )

            if hasattr(model, "classes_"):
                df_probs = pd.DataFrame({
                    "Sentiment": list(model.classes_),
                    "Probabilité (%)": [round(float(p) * 100, 1) for p in probs],
                })
                st.dataframe(df_probs, use_container_width=True, hide_index=True)
        else:
            st.success(f"**Sentiment détecté :** {pred}")
    else:
        df_batch = predire_liste(lignes)
        st.success(f"{len(lignes)} réclamations analysées.")
        st.dataframe(df_batch, use_container_width=True, hide_index=True)
        afficher_stats_sentiment(df_batch, "Sentiment")

st.divider()

# 3. Analyse d'un fichier CSV complet (upload)
st.subheader("📁 Importer et analyser un fichier CSV")
fichier_csv = st.file_uploader(
    "Choisissez un fichier CSV contenant les réclamations", type=["csv"]
)

if fichier_csv is not None:
    try:
        df_upload = pd.read_csv(fichier_csv, encoding="utf-8")
    except UnicodeDecodeError:
        fichier_csv.seek(0)
        df_upload = pd.read_csv(fichier_csv, encoding="latin-1")

    colonnes_texte = [
        c for c in ["texte", "reclamation", "commentaire", "message"]
        if c in df_upload.columns
    ]
    if not colonnes_texte:
        colonnes_texte = list(df_upload.columns)

    col_choisie = st.selectbox(
        "Sélectionnez la colonne contenant le texte des réclamations :",
        options=colonnes_texte,
    )

    if st.button("Lancer l'analyse du fichier CSV"):
        textes_csv = df_upload[col_choisie].fillna("").astype(str).tolist()
        df_res_csv = df_upload.copy()
        df_res_csv["Sentiment_predit"] = model.predict(textes_csv)

        if hasattr(model, "predict_proba"):
            probs_csv = model.predict_proba(textes_csv)
            df_res_csv["Confiance (%)"] = [
                round(float(max(p)) * 100, 1) for p in probs_csv
            ]

        st.success(
            f"Analyse terminée sur {len(df_res_csv)} lignes."
        )
        st.dataframe(df_res_csv, use_container_width=True, hide_index=True)
        afficher_stats_sentiment(df_res_csv, "Sentiment_predit")

        csv_export = df_res_csv.to_csv(index=False, encoding="utf-8").encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger les résultats (CSV)",
            data=csv_export,
            file_name="reclamations_analysees.csv",
            mime="text/csv",
        )

st.divider()

# 4. Statistiques globales du dataset local (data/reclamations.csv)
st.subheader("📊 Statistiques du dataset local (`data/reclamations.csv`)")

if os.path.exists(DATA_PATH):
    df_local = pd.read_csv(DATA_PATH, encoding="utf-8")
    col_locale = (
        "texte"
        if "texte" in df_local.columns
        else ("reclamation" if "reclamation" in df_local.columns else None)
    )

    if col_locale is not None and len(df_local) > 0:
        textes_locaux = df_local[col_locale].fillna("").astype(str).tolist()
        if "sentiment" not in df_local.columns:
            df_local["sentiment"] = model.predict(textes_locaux)

        afficher_stats_sentiment(df_local, "sentiment")
        st.dataframe(df_local.head(20), use_container_width=True, hide_index=True)
    else:
        st.info(
            "Le fichier `data/reclamations.csv` est vide ou ne contient pas de"
            " colonne texte (`texte` / `reclamation`)."
        )
else:
    st.warning("Fichier `data/reclamations.csv` introuvable.")
