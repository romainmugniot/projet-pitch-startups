"""Collecte : appels aux API Y Combinator et Hacker News, sauvegarde et relecture sur S3.

Responsable : membre 1.
"""

import pandas as pd
import requests

from src.config import URL_YC 


def telecharger_yc(url: str = URL_YC) -> pd.DataFrame:
    """Télécharge la base complète des startups Y Combinator.

    Renvoie un DataFrame avec une ligne par startup et les colonnes
    d'origine de l'API (name, slug, batch, status, one_liner, ...).
    """
    reponse = requests.get(url, timeout=60)
    reponse.raise_for_status()  # arrête tout avec un message clair si le site répond par une erreur

    startups = pd.DataFrame(reponse.json())

    if startups.empty:
        raise ValueError("La base YC téléchargée est vide")
    colonnes_attendues = {"slug", "name", "batch", "status", "one_liner"}
    manquantes = colonnes_attendues - set(startups.columns)
    if manquantes:
        raise ValueError(f"Colonnes absentes de l'API YC : {manquantes}")

    return startups



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
