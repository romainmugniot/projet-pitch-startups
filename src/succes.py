"""Succès : construction de la variable à prédire.

Responsable : membre 3.
"""

import pandas as pd


def niveau_succes(df: pd.DataFrame, seuil_equipe: int) -> pd.Series:
    """Classe chaque startup en quatre niveaux.

    - échec : inactive ;
    - survie : active, stade Early ;
    - croissance : active, stade Growth ou équipe d'au moins `seuil_equipe` personnes ;
    - sortie : rachetée ou en bourse.
    """
    raise NotImplementedError


def succes_binaire(niveau: pd.Series) -> pd.Series:
    """1 si la startup est active, rachetée ou en bourse, 0 si elle est inactive."""
    raise NotImplementedError


def rachats_de_sauvetage(df: pd.DataFrame, taille_max: int) -> pd.Series:
    """Repère les rachats probablement « de sauvetage » (rachetées avec une très petite équipe)."""
    raise NotImplementedError


def controles_qualite(df: pd.DataFrame) -> pd.DataFrame:
    """Statuts manquants, équipes à 0, doublons, promotions mal renseignées."""
    raise NotImplementedError
