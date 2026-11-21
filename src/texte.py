"""Texte : nettoyage des pitchs et indicateurs simples.

Responsable : membre 2. Toutes les fonctions acceptent n'importe quelle colonne de pitchs
(`pitch_actuel` ou `pitch_origine`).
"""

import re
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Liste de buzzwords à justifier dans le rapport : termes marketing génériques ou effets de mode
# qui décrivent peu le produit réel ou le client ciblé.
BUZZWORDS: list[str] = [
    "ai",
    "ai-powered",
    "all-in-one",
    "autonomous",
    "blockchain",
    "cloud-native",
    "crypto",
    "cutting-edge",
    "democratize",
    "disrupt",
    "disrupting",
    "disruptive",
    "ecosystem",
    "empower",
    "empowering",
    "end-to-end",
    "enterprise-grade",
    "future",
    "hyper",
    "innovative",
    "intelligent",
    "leverage",
    "metaverse",
    "next-gen",
    "nextgen",
    "paradigm",
    "powered",
    "reimagine",
    "reinvent",
    "reinventing",
    "revolutionary",
    "scalable",
    "seamless",
    "smart",
    "supercharge",
    "synergy",
    "transform",
    "transformative",
    "web3",
    "world-class",
]

# Expressions régulières compilées une seule fois pour les indicateurs structurels.
_MOTIF_CHIFFRE = re.compile(r"\d")
_MOTIF_X_FOR_Y = re.compile(
    r"\b(?:the\s+[a-z0-9\-]+\s+(?:for|of)\s+[a-z0-9]"
    r"|(?:uber|airbnb|stripe|aws|amazon|netflix|shopify|github|slack|figma|google|kayak|tinder"
    r"|salesforce|linkedin|spotify|paypal|dropbox|twilio|plaid|canva)\s+(?:for|of)\b)",
    re.IGNORECASE,
)
_MOTIF_FOR_CIBLE = re.compile(r"\bfor\s+([a-z][a-z\-]*)", re.IGNORECASE)
_FAUSSES_CIBLES = {
    "a",
    "all",
    "an",
    "any",
    "anyone",
    "better",
    "everyone",
    "fast",
    "faster",
    "free",
    "good",
    "life",
    "modern",
    "more",
    "new",
    "now",
    "people",
    "real",
    "the",
    "their",
    "this",
    "your",
}


@lru_cache(maxsize=1)
def _ressources_nlp() -> tuple[frozenset[str], WordNetLemmatizer]:
    """Charge les stopwords anglais et le lemmatiseur WordNet (télécharge si absents)."""
    for chemin, paquet in [("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(chemin)
        except LookupError:
            nltk.download(paquet, quiet=True)
    return frozenset(stopwords.words("english")), WordNetLemmatizer()


def _texte_propre_brut(pitch: object) -> str:
    """Convertit une entrée potentiellement manquante en chaîne propre."""
    if pitch is None or pd.isna(pitch):
        return ""
    return str(pitch).strip()


def nettoyer_pitch(pitch: str) -> str:
    """Minuscules, sans ponctuation ni mots vides anglais, lemmatisé.

    Renvoie une chaîne vide si `pitch` est manquant ou vide.
    """
    texte = _texte_propre_brut(pitch).lower()
    if not texte:
        return ""

    mots_vides, lemmatiseur = _ressources_nlp()
    tokens = re.findall(r"[a-z]+", texte)
    tokens_propres = [
        lemmatiseur.lemmatize(mot) for mot in tokens if mot not in mots_vides and len(mot) > 1
    ]
    return " ".join(tokens_propres)


def _a_client_nomme(texte: str) -> bool:
    """Détecte la présence d'un segment client explicite après « for » (« for doctors », etc.)."""
    for match in _MOTIF_FOR_CIBLE.finditer(texte):
        mot_suivant = match.group(1).lower()
        if mot_suivant not in _FAUSSES_CIBLES:
            return True
    return False


def indicateurs(pitchs: pd.Series) -> pd.DataFrame:
    """Indicateurs par pitch : longueur, présence de chiffres, tournure « the X for Y »
    ou « Uber for », client nommé (« for doctors », « for restaurants ») et score de jargon.

    Accepte n'importe quelle `Series` de pitchs et renvoie un DataFrame aligné sur son index.
    """
    textes = pitchs.map(_texte_propre_brut)
    return pd.DataFrame(
        {
            "nb_mots": textes.map(lambda t: len(t.split())),
            "nb_caracteres": textes.map(len),
            "a_chiffre": textes.map(lambda t: bool(_MOTIF_CHIFFRE.search(t))),
            "tournure_x_for_y": textes.map(lambda t: bool(_MOTIF_X_FOR_Y.search(t))),
            "client_nomme": textes.map(_a_client_nomme),
            "score_jargon": textes.map(score_jargon),
        },
        index=pitchs.index,
    )


def score_jargon(pitch: str) -> float:
    """Part des mots du pitch qui figurent dans `BUZZWORDS` (entre 0.0 et 1.0)."""
    texte = _texte_propre_brut(pitch).lower()
    if not texte:
        return 0.0

    # On conserve les tirets internes pour reconnaître « ai-powered » ou « end-to-end ».
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", texte)
    if not tokens:
        return 0.0

    ensemble_buzzwords = set(BUZZWORDS)
    nb_buzzwords = sum(1 for mot in tokens if mot in ensemble_buzzwords)
    return nb_buzzwords / len(tokens)


def mots_associes(pitchs: pd.Series, succes: pd.Series, freq_min: int = 10) -> pd.DataFrame:
    """Mots les plus associés au succès et à l'échec, en écartant ceux vus moins de `freq_min` fois.

    `succes` doit être une série binaire (1 = succès, 0 = échec) alignée sur `pitchs`.
    Renvoie un DataFrame indexé par mot, trié par `log_odds_ratio` décroissant, avec les colonnes :
    `frequence`, `nb_succes`, `nb_echec`, `taux_succes`, `ecart_taux`, `log_odds_ratio`.
    """
    donnees = pd.DataFrame({"pitch": pitchs, "succes": succes}).dropna(subset=["succes"])
    y = donnees["succes"].astype(int)
    n_succes_total = int((y == 1).sum())
    n_echec_total = int((y == 0).sum())
    if n_succes_total == 0 or n_echec_total == 0:
        raise ValueError("La série `succes` doit contenir au moins un succès et un échec.")

    taux_global = n_succes_total / (n_succes_total + n_echec_total)

    # Comptage par présence dans le document (chaque mot compte au plus une fois par startup).
    compteurs: dict[str, list[int]] = {}
    for texte, est_succes in zip(donnees["pitch"], y, strict=True):
        mots_uniques = set(nettoyer_pitch(texte).split())
        for mot in mots_uniques:
            if mot not in compteurs:
                compteurs[mot] = [0, 0]
            compteurs[mot][est_succes] += 1

    lignes = []
    for mot, (nb_echec, nb_succes) in compteurs.items():
        frequence = nb_succes + nb_echec
        if frequence < freq_min:
            continue
        taux_succes = nb_succes / frequence
        # Log-odds ratio avec lissage de Laplace (+0.5) pour éviter les divisions par zéro.
        odds_succes = (nb_succes + 0.5) / (n_succes_total - nb_succes + 0.5)
        odds_echec = (nb_echec + 0.5) / (n_echec_total - nb_echec + 0.5)
        lignes.append(
            {
                "mot": mot,
                "frequence": frequence,
                "nb_succes": nb_succes,
                "nb_echec": nb_echec,
                "taux_succes": taux_succes,
                "ecart_taux": taux_succes - taux_global,
                "log_odds_ratio": float(np.log(odds_succes / odds_echec)),
            }
        )

    colonnes = [
        "frequence",
        "nb_succes",
        "nb_echec",
        "taux_succes",
        "ecart_taux",
        "log_odds_ratio",
    ]
    if not lignes:
        return pd.DataFrame(columns=colonnes).rename_axis("mot")

    return (
        pd.DataFrame(lignes)
        .set_index("mot")
        .sort_values("log_odds_ratio", ascending=False)[colonnes]
    )


def similarite_pitchs(pitch_origine: pd.Series, pitch_actuel: pd.Series) -> pd.Series:
    """Similarité de Jaccard entre le vocabulaire nettoyé du pitch d'origine et du pitch actuel.

    Renvoie 1.0 si les deux pitchs contiennent exactement les mêmes mots significatifs, 0.0 s'ils
    n'ont aucun mot en commun, et `NaN` si l'un des deux pitchs est vide.
    """
    scores = []
    for origine, actuel in zip(pitch_origine, pitch_actuel, strict=True):
        mots_origine = set(nettoyer_pitch(origine).split())
        mots_actuel = set(nettoyer_pitch(actuel).split())
        if not mots_origine or not mots_actuel:
            scores.append(np.nan)
        else:
            scores.append(len(mots_origine & mots_actuel) / len(mots_origine | mots_actuel))
    return pd.Series(scores, index=pitch_origine.index, name="similarite_pitch")
