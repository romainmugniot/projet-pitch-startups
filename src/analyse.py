"""Analyse : taux d'échec, comparaisons à promotion égale et réaction sur Hacker News."""

from statistics import NormalDist

import numpy as np
import pandas as pd

from src.succes import echec

# Quantile de la loi normale pour les intervalles de confiance à 95 % (1,96).
Z_95 = NormalDist().inv_cdf(0.975)

AUTRES = "Autres"
NON_RENSEIGNE = "Non renseigné"


def _modalites(df: pd.DataFrame, par: str, min_effectif: int) -> pd.Series:
    """Valeurs de la colonne `par`, prêtes à comparer.

    Les valeurs manquantes deviennent « Non renseigné » et les modalités de moins de
    `min_effectif` startups sont regroupées en « Autres » : un taux calculé sur trois startups
    ne dit presque rien.
    """
    valeurs = df[par].astype("object").fillna(NON_RENSEIGNE)
    effectifs = valeurs.map(valeurs.value_counts())
    return valeurs.where(effectifs >= min_effectif, AUTRES)


def _trier(resultat: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Trie du plus grand au plus petit sur `colonne`, « Autres » et « Non renseigné » en dernier."""
    resultat = resultat.sort_values(colonne, ascending=False)
    en_dernier = resultat.index.isin([AUTRES, NON_RENSEIGNE])
    return pd.concat([resultat[~en_dernier], resultat[en_dernier]])


def _intervalle_wilson(
    nombre: pd.Series, effectif: pd.Series, z: float = Z_95
) -> tuple[pd.Series, pd.Series]:
    """Bornes de l'intervalle de Wilson (95 % par défaut) de la proportion `nombre / effectif`.

    Plus fiable que « taux ± 1,96 écart-type » sur de petits effectifs ou des taux proches de 0 :
    ses bornes restent toujours entre 0 et 1.
    """
    p = nombre / effectif
    centre = (p + z**2 / (2 * effectif)) / (1 + z**2 / effectif)
    demi_largeur = (
        z / (1 + z**2 / effectif) * np.sqrt(p * (1 - p) / effectif + z**2 / (4 * effectif**2))
    )
    return (centre - demi_largeur).clip(lower=0), (centre + demi_largeur).clip(upper=1)


def _intervalle_poisson(observes: pd.Series, z: float = Z_95) -> tuple[pd.Series, pd.Series]:
    """Bornes de l'intervalle à 95 % d'un nombre d'échecs supposé suivre une loi de Poisson.

    Approximation de Byar : les nombres d'échecs « vrais » compatibles avec les `observes` échecs
    comptés. Précise dès quelques échecs ; pour 0 échec, la borne basse vaut 0.
    """
    o = observes.astype(float)
    bas = o * (1 - 1 / (9 * o) - z / (3 * np.sqrt(o))) ** 3
    haut = (o + 1) * (1 - 1 / (9 * (o + 1)) + z / (3 * np.sqrt(o + 1))) ** 3
    return bas.where(o > 0, 0.0), haut


def taux_echec(df: pd.DataFrame, par: str, min_effectif: int = 10) -> pd.DataFrame:
    """Taux d'échec par modalité de `par` (secteur, région…), avec son intervalle de confiance.

    Une ligne par modalité : `effectif`, `echecs` (au sens de `succes.echec`), `taux`, et les
    bornes `ic_bas` et `ic_haut` de l'intervalle de Wilson à 95 %. Les modalités de moins de
    `min_effectif` startups sont regroupées en « Autres », les valeurs manquantes en
    « Non renseigné ». Lignes triées du taux le plus haut au plus bas, ces deux groupes en dernier.
    """
    groupes = _modalites(df, par, min_effectif)
    resultat = echec(df).groupby(groupes).agg(effectif="size", echecs="sum")
    resultat["taux"] = resultat["echecs"] / resultat["effectif"]
    resultat["ic_bas"], resultat["ic_haut"] = _intervalle_wilson(
        resultat["echecs"], resultat["effectif"]
    )
    return _trier(resultat, "taux")


def taux_echec_standardise(
    df: pd.DataFrame, par: str, min_effectif: int = 10, strates: tuple[str, ...] = ("promotion",)
) -> pd.DataFrame:
    """Échecs observés rapportés aux échecs attendus à promotion égale, par modalité de `par`.

    Une promotion récente a eu moins de temps pour échouer : comparer des taux bruts reviendrait
    en partie à comparer des âges. Chaque startup reçoit donc le taux d'échec de sa promotion ;
    leur somme sur une modalité donne les échecs « attendus » si ses startups échouaient comme
    leurs promotions (standardisation indirecte).

    Une ligne par modalité (regroupées comme dans `taux_echec`) : `effectif`, `observes`,
    `attendus`, `ratio` = observés / attendus, et son intervalle à 95 % (`ic_bas`, `ic_haut`),
    en supposant que le nombre d'échecs suit une loi de Poisson. Un ratio de 1,3 se lit
    « 30 % d'échecs de plus que leurs promotions », 0,8 « 20 % de moins » ; si l'intervalle
    contient 1, l'écart peut venir du hasard.

    `strates` fixe les cases dont chaque startup reçoit le taux d'échec : la promotion par défaut.
    Avec `strates=("promotion", "secteur")`, chaque startup reçoit le taux de sa case
    promotion × secteur, et le ratio se lit « à promotion et secteur égaux ». Une case d'une
    seule startup a autant d'échecs attendus qu'observés : elle n'apporte rien à la comparaison.
    """
    echecs = echec(df)
    cases = [df[strate].astype("object").fillna(NON_RENSEIGNE) for strate in strates]
    attendus = echecs.groupby(cases).transform("mean")
    groupes = _modalites(df, par, min_effectif)
    resultat = (
        pd.DataFrame({"echec": echecs, "attendu": attendus})
        .groupby(groupes)
        .agg(effectif=("echec", "size"), observes=("echec", "sum"), attendus=("attendu", "sum"))
    )
    resultat["ratio"] = resultat["observes"] / resultat["attendus"]
    bas, haut = _intervalle_poisson(resultat["observes"])
    resultat["ic_bas"] = bas / resultat["attendus"]
    resultat["ic_haut"] = haut / resultat["attendus"]
    return _trier(resultat, "ratio")


def ratios_par_indicateur(
    df: pd.DataFrame, colonnes: list[str], strates: tuple[str, ...] = ("promotion",)
) -> pd.DataFrame:
    """Échecs observés / attendus à promotion égale pour chaque indicateur booléen de `colonnes`.

    Pour chaque colonne (« le pitch contient un chiffre »…), garde la ligne des startups pour
    lesquelles l'indicateur est vrai dans `taux_echec_standardise`, sans regrouper les petits
    effectifs : un indicateur rare garde sa ligne, avec un intervalle large. Une ligne par
    indicateur, dans l'ordre de `colonnes` ; un indicateur jamais vrai est omis. `strates` est
    transmis à `taux_echec_standardise` (promotion × secteur pour comparer à secteur égal aussi).
    """
    lignes = {}
    for colonne in colonnes:
        resultat = taux_echec_standardise(
            df.assign(**{colonne: df[colonne].astype(bool)}), colonne, 1, strates
        )
        if True in resultat.index:
            lignes[colonne] = resultat.loc[True]
    resultat = pd.DataFrame(lignes).T.rename_axis("indicateur")
    return resultat.astype({"effectif": int, "observes": int})


def reaction_hn_et_destin(df: pd.DataFrame) -> pd.DataFrame:
    """Réaction sur Hacker News (points, commentaires) selon le destin de la startup.

    Une ligne par statut (rachetées, actives, inactives, puis les autres) : `startups`, puis le
    premier quartile, la médiane et le troisième quartile des points (`points_q1`,
    `points_mediane`, `points_q3`) et des commentaires (`commentaires_q1`…). Les quartiles plutôt
    que la moyenne : quelques posts à plus de 1 000 points tireraient la moyenne vers le haut.
    """
    groupes = df.groupby("statut")
    resultat = groupes.size().to_frame("startups")
    for colonne, nom in [("hn_points", "points"), ("hn_commentaires", "commentaires")]:
        resultat[f"{nom}_q1"] = groupes[colonne].quantile(0.25)
        resultat[f"{nom}_mediane"] = groupes[colonne].median()
        resultat[f"{nom}_q3"] = groupes[colonne].quantile(0.75)
    ordre = ["Acquired", "Active", "Inactive"]
    return resultat.loc[
        [s for s in ordre if s in resultat.index] + [s for s in resultat.index if s not in ordre]
    ]
