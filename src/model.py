import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import tokeniser


def construire_modele() -> Pipeline:
    """Pipeline TF-IDF (unigrammes + bigrammes) + Régression Logistique."""
    return Pipeline(
        [
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
        ]
    )


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


def evaluer_modele(X_test: pd.Series, y_test: pd.Series, y_pred) -> None:
    """Calcule et affiche les métriques d'évaluation et sauvegarde la matrice de confusion."""
    print("\n--- Évaluation sur le jeu de test (20%) ---")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("\nRapport détaillé (Précision, Rappel, F1-score) :")
    print(classification_report(y_test, y_pred, digits=3))

    labels = ["Positif", "Neutre", "Négatif"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Matrice de confusion :\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion - Jeu de test")
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=200)
    plt.close()
    print("Matrice de confusion sauvegardée dans : matrice_confusion.png")


def entrainer_modele(
    csv_path: str = "data/reclamations.csv",
    model_dir: str = "models",
    test_size: float = 0.2,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=["reclamation", "sentiment"])

    X = df["reclamation"]
    y = df["sentiment"]

    # 1. Évaluation sur un jeu de test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    eval_model = construire_modele()
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    evaluer_modele(X_test, y_test, y_pred)

    # 2. Entraînement final sur l'ensemble des données pour le déploiement
    model = construire_modele()
    model.fit(X, y)

    preds = model.predict(X)
    accuracy_train = sum(p == y_true for p, y_true in zip(preds, y)) / len(df)
    print(f"\nNombre de réclamations : {len(df)}")
    print(f"Précision (train) : {accuracy_train * 100:.1f}%")

    afficher_top_coefficients(model, top_n=8)

    model_path = os.path.join(model_dir, "sentiment_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    entrainer_modele()
