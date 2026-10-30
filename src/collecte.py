"""Collecte : appels aux API Y Combinator et Hacker News, sauvegarde et relecture sur S3.

Responsable : membre 1.
"""

import datetime

import pandas as pd
import requests
import s3fs

from src.config import S3_ENDPOINT, URL_HN, URL_YC

COLONNES_HN = [
    "objectID",
    "title",
    "url",
    "author",
    "created_at",
    "created_at_i",
    "points",
    "num_comments",
    "story_text",
]


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


def telecharger_launch_hn(url: str = URL_HN) -> pd.DataFrame:
    """Récupère tous les posts Launch HN (tag `launch_hn`) de l'API Algolia.

    L'API renvoie au plus 1 000 résultats par requête : on demande les posts du plus récent
    au plus ancien, puis on relance la recherche avant la date du plus ancien reçu, jusqu'à
    ce qu'il n'y en ait plus. Renvoie un DataFrame avec une ligne par post et les colonnes
    d'origine de l'API (objectID, title, created_at, points, num_comments, ...).
    """
    parametres = {
        "tags": "launch_hn",
        "hitsPerPage": 1000,
        "attributesToRetrieve": ",".join(COLONNES_HN),
        "attributesToHighlight": "",
    }
    posts = []
    while True:
        reponse = requests.get(url, params=parametres, timeout=60)
        reponse.raise_for_status()
        resultats = reponse.json()["hits"]
        if not resultats:
            break
        posts.extend(resultats)
        plus_ancien = resultats[-1]["created_at_i"]
        parametres["numericFilters"] = f"created_at_i<{plus_ancien}"

    if not posts:
        raise ValueError("Aucun post Launch HN téléchargé")

    return pd.DataFrame(posts, columns=COLONNES_HN)


def connexion_s3() -> s3fs.S3FileSystem:
    """Ouvre la connexion au S3 du SSP Cloud.

    Les identifiants (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN) sont lus
    dans les variables d'environnement, déjà renseignées dans les services du SSP Cloud.
    """
    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})


def sauvegarder_s3(df: pd.DataFrame, nom: str, dossier: str) -> str:
    """Sauvegarde `df` en Parquet daté sur le S3 du SSP Cloud et renvoie le chemin écrit.

    Le fichier est rangé sous `dossier/nom/AAAA-MM-JJ.parquet`, par exemple
    `CHEMIN_BRUT/yc/2026-10-27.parquet` : une copie par jour, celle du jour remplace la
    précédente si on sauvegarde deux fois.
    """
    chemin = f"{dossier}/{nom}/{datetime.date.today().isoformat()}.parquet"
    with connexion_s3().open(chemin, "wb") as fichier:
        df.to_parquet(fichier, index=False)
    return chemin


def lire_s3(nom: str, dossier: str, date: str | None = None) -> pd.DataFrame:
    """Relit une copie Parquet depuis le S3 (la plus récente si `date` est None).

    `date` s'écrit « AAAA-MM-JJ », comme dans le nom des fichiers écrits par `sauvegarder_s3`.
    """
    s3 = connexion_s3()
    if date is None:
        copies = sorted(s3.glob(f"{dossier}/{nom}/*.parquet"))
        if not copies:
            raise FileNotFoundError(f"Aucune copie de « {nom} » dans {dossier}")
        chemin = copies[-1]  # les dates AAAA-MM-JJ se trient dans l'ordre chronologique
    else:
        chemin = f"{dossier}/{nom}/{date}.parquet"

    with s3.open(chemin, "rb") as fichier:
        return pd.read_parquet(fichier)
