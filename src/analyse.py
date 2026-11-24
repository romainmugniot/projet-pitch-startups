"""Analyse : statistiques descriptives et modèles.

Responsables : membre 3 (statistiques), membre 2 (modèles).
"""

import pandas as pd


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


def modele_reference():
    """Pipeline scikit-learn : régression logistique sur secteur, promotion et région."""
    raise NotImplementedError


def modele_texte():
    """Même pipeline que `modele_reference`, plus le pitch transformé en TF-IDF."""
    raise NotImplementedError


def evaluer(modele, X_test, y_test) -> dict:
    """AUC et matrice de confusion, en tenant compte du déséquilibre succès / échec."""
    raise NotImplementedError
