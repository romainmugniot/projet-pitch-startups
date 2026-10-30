"""Nettoyage : extraction des titres Launch HN, harmonisation des noms, jointure avec la base YC.

Responsable : membre 1.
"""

import re

import pandas as pd
from unidecode import unidecode

from src.config import COLONNES_TABLE_COMMUNE

# « Launch HN: Nom (YC W20) – pitch ». Après la promotion, le séparateur est facultatif
# (« Nom (YC S17), pitch » existe). Sans promotion, le pitch doit être séparé du nom par un
# tiret long (–, —), deux-points ou un tiret court entouré d'espaces.
MOTIF_TITRE = re.compile(
    r"^Launch HN\s*[:\-–—]?\s*(?P<nom>.+?)\s*"
    r"(?:\(YC (?P<saison>[WSFXP])(?P<annee>\d{2})[^)]*\)\s*[,:\-–—]?\s*|\s*[–—:]\s*|\s+-\s+)"
    r"(?P<pitch>.+)$"
)

# Lettres utilisées par YC dans les titres. X et P désignent toutes deux la promotion de printemps.
SAISONS = {"W": "Winter", "S": "Summer", "F": "Fall", "X": "Spring", "P": "Spring"}

# Mots ajoutés ou retirés du nom selon la source (« Scanwell » / « Scanwell Health »).
SUFFIXES = {"inc", "co", "ai", "io", "app", "labs", "hq", "technologies", "software", "health"}

# Grandes régions de la base YC, plus utiles que le pays pour comparer des groupes de taille suffisante.
REGIONS = [
    "America / Canada",
    "Europe",
    "South Asia",
    "Latin America",
    "Southeast Asia",
    "Middle East and North Africa",
    "Africa",
    "East Asia",
    "Oceania",
]


def extraire_titre(titre: str) -> dict:
    """Découpe un titre « Launch HN: Nom (YC W20) – pitch ».

    Renvoie un dict {"nom", "promotion", "pitch"} ; « W20 » devient « Winter 2020 ».
    Gère les variantes de tiret (-, –, —) et les titres sans promotion (promotion = None).
    Si le titre ne suit pas ce format, les trois valeurs sont None.
    """
    resultat = MOTIF_TITRE.match(titre.strip())
    if resultat is None:
        return {"nom": None, "promotion": None, "pitch": None}

    promotion = None
    if resultat["saison"]:
        promotion = f"{SAISONS[resultat['saison']]} 20{resultat['annee']}"
    return {"nom": resultat["nom"], "promotion": promotion, "pitch": resultat["pitch"].strip()}


def normaliser_nom(nom: str) -> str:
    """Minuscules, sans accents ni ponctuation, sans suffixes (« Inc », « AI », « Labs », « HQ »,
    « .io », « Technologies »… : voir `SUFFIXES`).

    Les mots sont recollés sans espace (« Better Auth » et « BetterAuth » donnent « betterauth »).
    Un suffixe n'est retiré que s'il reste un autre mot (« Labs » seul reste « labs »).
    """
    mots = re.sub(r"[^a-z0-9]+", " ", unidecode(nom).lower()).split()
    while len(mots) > 1 and mots[-1] in SUFFIXES:
        mots.pop()
    return "".join(mots)


def _grande_region(regions: list) -> str | None:
    """Première grande région (`REGIONS`) de la liste des régions YC, None s'il n'y en a pas."""
    return next((region for region in regions if region in REGIONS), None)


def joindre(yc: pd.DataFrame, hn: pd.DataFrame) -> pd.DataFrame:
    """Apparie les posts Launch HN à la base YC.

    Première passe sur le nom normalisé et la promotion, seconde passe sur les anciens noms
    pour les startups renommées. Renvoie la table commune (`config.COLONNES_TABLE_COMMUNE`).

    Une ligne par startup appariée : si une startup a plusieurs posts Launch HN, on garde le
    plus ancien, qui porte son pitch d'origine. Les posts sans promotion ne sont pas appariés,
    pour éviter de confondre deux startups homonymes. `niveau_succes` est laissé vide : il est
    calculé ensuite par `succes.niveau_succes`.
    """
    posts = pd.concat(
        [hn, pd.DataFrame(hn["title"].map(extraire_titre).tolist(), index=hn.index)], axis=1
    )
    posts = posts.dropna(subset=["nom", "promotion"])
    posts["cle"] = posts["nom"].map(normaliser_nom)

    startups = yc.assign(cle=yc["name"].map(normaliser_nom))
    par_nom = posts.merge(startups, left_on=["cle", "promotion"], right_on=["cle", "batch"])

    anciens_noms = startups[["slug", "batch", "former_names"]].explode("former_names").dropna()
    anciens_noms["cle"] = anciens_noms["former_names"].map(normaliser_nom)
    restants = posts[~posts["objectID"].isin(par_nom["objectID"])]
    par_ancien_nom = restants.merge(
        anciens_noms[["slug", "batch", "cle"]],
        left_on=["cle", "promotion"],
        right_on=["cle", "batch"],
    ).merge(startups.drop(columns=["cle", "batch"]), on="slug")

    apparies = (
        pd.concat([par_nom, par_ancien_nom], ignore_index=True)
        .drop_duplicates(subset="objectID")
        .sort_values("created_at_i")
        .drop_duplicates(subset="slug")
    )

    table = pd.DataFrame(
        {
            "slug": apparies["slug"],
            "nom": apparies["name"],
            "promotion": apparies["batch"],
            "pitch_origine": apparies["pitch"],
            "pitch_actuel": apparies["one_liner"],
            "statut": apparies["status"],
            "niveau_succes": pd.NA,
            "secteur": apparies["industry"],
            "region": apparies["regions"].map(_grande_region),
            "taille_equipe": apparies["team_size"],
            "anciens_noms": apparies["former_names"],
            "hn_date": pd.to_datetime(apparies["created_at"]),
            "hn_points": apparies["points"],
            "hn_commentaires": apparies["num_comments"],
        }
    )
    return table[COLONNES_TABLE_COMMUNE].reset_index(drop=True)


def bilan_jointure(table: pd.DataFrame, hn: pd.DataFrame) -> pd.DataFrame:
    """Nombre de posts, d'appariements et taux d'appariement par promotion.

    Les appariements comptent les startups de `table` : une startup lancée deux fois sur
    Hacker News compte pour deux posts mais un seul appariement. Les posts dont le titre
    n'indique pas de promotion sont comptés sous « inconnue ».
    Renvoie un DataFrame indexé par promotion, de la plus ancienne à la plus récente.
    """
    promotions = (
        hn["title"].map(lambda titre: extraire_titre(titre)["promotion"]).fillna("inconnue")
    )
    bilan = (
        pd.DataFrame(
            {
                "posts": promotions.value_counts(),
                "appariements": table["promotion"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    bilan["taux"] = bilan["appariements"] / bilan["posts"]

    ordre_saisons = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    return bilan.sort_index(
        key=lambda index: [
            (int(p.split()[1]), ordre_saisons[p.split()[0]]) if p != "inconnue" else (9999, 0)
            for p in index
        ]
    )
