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


def decoupage_temporel(df: pd.DataFrame, promotions_test: list[str]) -> tuple:
    """Entraînement sur les promotions anciennes, test sur les plus récentes."""
    raise NotImplementedError


def modele_reference():
    """Pipeline scikit-learn : régression logistique sur secteur, promotion et région."""
    raise NotImplementedError


def modele_texte():
    """Même pipeline que `modele_reference`, plus le pitch transformé en TF-IDF."""
    raise NotImplementedError


def evaluer(modele, X_test, y_test) -> dict:
    """AUC et matrice de confusion, en tenant compte du déséquilibre succès / échec."""
    raise NotImplementedError
