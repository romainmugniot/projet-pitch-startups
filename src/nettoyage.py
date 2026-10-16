"""Nettoyage : extraction des titres Launch HN, harmonisation des noms, jointure avec la base YC.

Responsable : membre 1.
"""

import pandas as pd


def extraire_titre(titre: str) -> dict:
    """Découpe un titre « Launch HN: Nom (YC W20) – pitch ».

    Renvoie un dict {"nom", "promotion", "pitch"} ; « W20 » devient « Winter 2020 ».
    Gère les variantes de tiret (-, –, —) et les titres sans promotion (promotion = None).
    """
    raise NotImplementedError


def normaliser_nom(nom: str) -> str:
    """Minuscules, sans accents ni ponctuation, sans suffixes (« Inc », « AI », « Labs », « HQ »)."""
    raise NotImplementedError


def joindre(yc: pd.DataFrame, hn: pd.DataFrame) -> pd.DataFrame:
    """Apparie les posts Launch HN à la base YC.

    Première passe sur le nom normalisé et la promotion, seconde passe sur les anciens noms
    pour les startups renommées. Renvoie la table commune (`config.COLONNES_TABLE_COMMUNE`).
    """
    raise NotImplementedError


def bilan_jointure(table: pd.DataFrame, hn: pd.DataFrame) -> pd.DataFrame:
    """Nombre de posts, d'appariements et taux d'appariement par promotion."""
    raise NotImplementedError
