"""Graphiques homogènes : titre, axes nommés, légende, mêmes couleurs partout.

Responsable : membre 3.
"""

import pandas as pd

COULEURS = {"succes": "#1b7f5b", "echec": "#c0392b", "neutre": "#7f8c8d"}


def barres_taux_echec(taux: pd.DataFrame, titre: str):
    """Barres horizontales du taux d'échec par modalité (secteur, région)."""
    raise NotImplementedError


def courbe_par_promotion(taux: pd.DataFrame, titre: str):
    """Taux d'échec par promotion ; les promotions récentes ont moins de recul."""
    raise NotImplementedError


def nuages_compares(pitchs_succes: pd.Series, pitchs_echec: pd.Series):
    """Deux nuages de mots côte à côte : succès et échecs."""
    raise NotImplementedError


def mots_frequence_association(mots: pd.DataFrame):
    """Nuage de points : fréquence d'un mot contre son association au succès."""
    raise NotImplementedError
