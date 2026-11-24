"""Analyse : statistiques descriptives et modèles.

Responsables : membre 3 (statistiques), membre 2 (modèles).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.texte import nettoyer_pitch

COLONNES_STRUCTURELLES = ["secteur", "promotion", "region"]


def taux_echec(df: pd.DataFrame, par: str) -> pd.DataFrame:
    """Taux d'échec et effectifs par modalité de `par` (promotion, secteur, région)."""
    raise NotImplementedError


def reaction_hn_et_destin(df: pd.DataFrame) -> pd.DataFrame:
    """Lien entre la réaction sur Hacker News (points, commentaires) et le destin de la startup."""
    raise NotImplementedError


def decoupage_temporel(
    df: pd.DataFrame,
    promotions_test: list[str],
    cible: str = "succes",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Entraînement sur les promotions anciennes, test sur les plus récentes.

    Si `cible` est absente de `df`, elle est construite depuis `statut` (1 si statut != "Inactive",
    0 sinon). Renvoie `(X_train, X_test, y_train, y_test)`.
    """
    masque_test = df["promotion"].isin(promotions_test)
    if masque_test.all() or (~masque_test).all():
        raise ValueError(
            "Le découpage temporel doit contenir au moins une observation en train et en test."
        )

    if cible in df.columns:
        y = df[cible].astype(int)
        X = df.drop(columns=[cible])
    else:
        y = (df["statut"] != "Inactive").astype(int)
        X = df.copy()

    return (
        X.loc[~masque_test].reset_index(drop=True),
        X.loc[masque_test].reset_index(drop=True),
        y.loc[~masque_test].reset_index(drop=True),
        y.loc[masque_test].reset_index(drop=True),
    )


def _serie_en_texte(serie: pd.Series) -> pd.Series:
    """Remplace les valeurs manquantes d'une colonne de texte par des chaînes vides."""
    return serie.fillna("").astype(str)


def _encodeur_structurel() -> Pipeline:
    """Imputation des modalités manquantes puis encodage one-hot robuste aux modalités inédites."""
    return Pipeline(
        [
            ("imputation", SimpleImputer(strategy="constant", fill_value="Inconnu")),
            ("encodage", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def modele_reference() -> Pipeline:
    """Pipeline scikit-learn : régression logistique sur secteur, promotion et région."""
    pretraitement = ColumnTransformer(
        transformers=[
            ("structure", _encodeur_structurel(), COLONNES_STRUCTURELLES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pretraitement", pretraitement),
            ("modele", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )


def modele_texte(
    colonne_pitch: str = "pitch_origine",
    min_df: int = 2,
    max_features: int = 1000,
) -> Pipeline:
    """Même pipeline que `modele_reference`, plus le pitch transformé en TF-IDF."""
    branche_texte = Pipeline(
        [
            ("nettoyage_na", FunctionTransformer(_serie_en_texte, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=nettoyer_pitch,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=max_features,
                ),
            ),
        ]
    )
    pretraitement = ColumnTransformer(
        transformers=[
            ("structure", _encodeur_structurel(), COLONNES_STRUCTURELLES),
            ("texte", branche_texte, colonne_pitch),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pretraitement", pretraitement),
            ("modele", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )


def evaluer(modele: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """AUC, PR-AUC, balanced accuracy, F1 par classe et matrice de confusion.

    Tient compte du déséquilibre succès / échec.
    """
    y_pred = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_echec": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "f1_succes": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "matrice_confusion": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }
