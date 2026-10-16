"""Texte : nettoyage des pitchs et indicateurs simples.

Responsable : membre 2. Toutes les fonctions acceptent n'importe quelle colonne de pitchs
(`pitch_actuel` ou `pitch_origine`).
"""

import pandas as pd

# Liste de buzzwords à justifier dans le rapport.
BUZZWORDS: list[str] = []


def nettoyer_pitch(pitch: str) -> str:
    """Minuscules, sans ponctuation ni mots vides anglais, lemmatisé."""
    raise NotImplementedError


def indicateurs(pitchs: pd.Series) -> pd.DataFrame:
    """Indicateurs par pitch : nombre de mots, présence de chiffres, tournure « the X for Y »
    ou « Uber for », client nommé (« for doctors », « for restaurants »)."""
    raise NotImplementedError


def score_jargon(pitch: str) -> float:
    """Part des mots du pitch qui figurent dans `BUZZWORDS`."""
    raise NotImplementedError


def mots_associes(pitchs: pd.Series, succes: pd.Series, freq_min: int) -> pd.DataFrame:
    """Mots les plus associés au succès et à l'échec, en écartant ceux vus moins de `freq_min` fois."""
    raise NotImplementedError
