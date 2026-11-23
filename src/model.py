import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing import tokeniser


def construire_modele() -> Pipeline:
    """Pipeline TF-IDF (unigrammes + bigrammes) + Régression Logistique."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=tokeniser,
                token_pattern=None,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                C=2.0,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def afficher_top_coefficients(pipeline: Pipeline, top_n: int = 8) -> None:
    """Affiche les mots/expressions avec les plus forts coefficients par sentiment."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    print("\n--- Mots les plus influents (Coefficients) ---")
    for idx, classe in enumerate(clf.classes_):
        coefs = clf.coef_[idx]
        top_indices = coefs.argsort()[-top_n:][::-1]
        top_mots = [f"{feature_names[i]} ({coefs[i]:.2f})" for i in top_indices]
        print(f"{classe} : {', '.join(top_mots)}")


def entrainer_modele(csv_path="data/reclamations.csv", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=["reclamation", "sentiment"])

    model = construire_modele()
    model.fit(df["reclamation"], df["sentiment"])

    preds = model.predict(df["reclamation"])
    accuracy = sum(p == y for p, y in zip(preds, df["sentiment"])) / len(df)
    print(f"Nombre de réclamations : {len(df)}")
    print(f"Précision (train) : {accuracy * 100:.1f}%")

    afficher_top_coefficients(model, top_n=8)

    model_path = os.path.join(model_dir, "sentiment_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    entrainer_modele()
