# Sujet 17 – Analyse Automatique du Sentiment des Réclamations

## 1. Description du Projet

Ce projet consiste à développer une solution d'Intelligence Artificielle capable d'analyser automatiquement des réclamations clients afin d'identifier le sentiment exprimé (**Positif**, **Négatif** ou **Neutre**) et de dégager les **tendances globales** (statistiques et graphiques) pour faciliter la prise de décision[cite: 1].

## 2. Fonctionnalités Clés _(Ce que fait l'application)_

**Prétraitement NLP :** Nettoyage et normalisation automatique des textes des réclamations avant l'analyse.
**Classification automatique :** Prédiction du sentiment (_Positif_, _Négatif_, _Neutre_) accompagnée d'un score de confiance grâce à un modèle préentraîné[cite: 1].
**Analyse des tendances globales :** Calcul des pourcentages par catégorie de sentiment et génération de graphiques (histogrammes)
**Interface interactive :** Application web permettant de tester une réclamation en temps réel ou d'importer un fichier de données.

## 3. Technologies Utilisées _(Les outils techniques)_

**Langage :** Python 3.x
**NLP & Machine Learning :** spacy, scikit-learn (TF-IDF + Régression Logistique)
**Sauvegarde du modèle :** joblib
**Traitement de données & Visualisation :** `pandas`, `numpy`, `matplotlib`
**Interface Web :** `streamlit`

## 4. Auteurs

**ouertatani eya**
**sai roua**
