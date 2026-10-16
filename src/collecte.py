"""Collecte : appels aux API Y Combinator et Hacker News, sauvegarde et relecture sur S3.

Responsable : membre 1.
"""

import pandas as pd


def telecharger_yc() -> pd.DataFrame:
    """Lit la base YC complète (API yc-oss, `config.URL_YC`) et la renvoie en DataFrame."""
    raise NotImplementedError


def telecharger_launch_hn() -> pd.DataFrame:
    """Récupère tous les posts Launch HN (tag `launch_hn`) page par page.

    L'API Algolia renvoie au plus 1 000 résultats par requête : découper la recherche
    par tranches de dates (`numericFilters=created_at_i>...`).
    """
    raise NotImplementedError


def sauvegarder_s3(df: pd.DataFrame, nom: str, dossier: str) -> str:
    """Sauvegarde `df` en Parquet daté sur le S3 du SSP Cloud et renvoie le chemin écrit."""
    raise NotImplementedError


def lire_s3(nom: str, dossier: str, date: str | None = None) -> pd.DataFrame:
    """Relit une copie Parquet depuis le S3 (la plus récente si `date` est None)."""
    raise NotImplementedError
