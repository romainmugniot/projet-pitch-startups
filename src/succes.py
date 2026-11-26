"""Succès : définition de l'échec, des niveaux de succès et contrôles de qualité de la table."""

import pandas as pd

from src.nettoyage import debut_promotion

# Du moins bon au meilleur : les comparaisons (<, >) et les tris suivent cet ordre.
NIVEAUX = ["échec", "survie", "croissance", "sortie"]


def echec(df: pd.DataFrame) -> pd.Series:
    """1 si la startup est inactive (statut « Inactive »), 0 sinon.

    C'est la variable étudiée dans tout le rapport : une startup active, rachetée ou en bourse
    n'a pas échoué.
    """
    return (df["statut"] == "Inactive").astype(int).rename("echec")


def niveau_succes(df: pd.DataFrame, seuil_equipe: float | None = None) -> pd.Series:
    """Classe chaque startup en quatre niveaux ordonnés : échec < survie < croissance < sortie.

    - échec : inactive ;
    - sortie : rachetée ou en bourse (« Acquired », « Public ») ;
    - croissance : active, au stade « Growth » ou avec une équipe de plus de `seuil_equipe`
      personnes ;
    - survie : les autres startups actives.

    Par défaut (`seuil_equipe=None`), le seuil est la taille d'équipe médiane des startups
    actives de la même promotion : une startup plus ancienne a eu plus de temps pour embaucher.
    Un statut inconnu donne une valeur manquante.
    """
    actives = df["statut"] == "Active"
    if seuil_equipe is None:
        seuil = df["taille_equipe"].where(actives).groupby(df["promotion"]).transform("median")
    else:
        seuil = pd.Series(seuil_equipe, index=df.index)
    en_croissance = (df["stage"] == "Growth") | (df["taille_equipe"] > seuil)

    niveau = pd.Series(pd.NA, index=df.index, dtype="object")
    niveau[actives] = "survie"
    niveau[actives & en_croissance] = "croissance"
    niveau[df["statut"] == "Inactive"] = "échec"
    niveau[df["statut"].isin(["Acquired", "Public"])] = "sortie"
    return niveau.astype(pd.CategoricalDtype(NIVEAUX, ordered=True)).rename("niveau_succes")


def rachats_de_sauvetage(df: pd.DataFrame, taille_max: int = 5) -> pd.Series:
    """True pour les startups rachetées avec une équipe de `taille_max` personnes ou moins.

    Un tel rachat ressemble plus souvent à un recrutement de l'équipe ou à une sortie de secours
    qu'à une réussite : il vaut mieux le signaler avant de compter tous les rachats comme des
    succès.
    """
    return ((df["statut"] == "Acquired") & (df["taille_equipe"] <= taille_max)).rename(
        "rachat_de_sauvetage"
    )


def controles_qualite(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par contrôle : nombre et part des startups concernées.

    Contrôles : statuts manquants, équipes manquantes ou à 0, `slug` en double, régions
    manquantes, promotions non reconnues (hors format « Saison AAAA »).
    """
    controles = {
        "Statut manquant": df["statut"].isna(),
        "Équipe manquante ou à 0": df["taille_equipe"].isna() | (df["taille_equipe"] == 0),
        "Slug en double": df["slug"].duplicated(keep=False),
        "Région manquante": df["region"].isna(),
        "Promotion non reconnue": df["promotion"].map(debut_promotion).isna(),
    }
    resultat = pd.DataFrame(
        {"nombre": {nom: int(masque.sum()) for nom, masque in controles.items()}}
    )
    resultat["part"] = resultat["nombre"] / len(df) if len(df) else 0.0
    resultat.index.name = "controle"
    return resultat
