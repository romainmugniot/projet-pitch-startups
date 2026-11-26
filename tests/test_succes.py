"""Cas attendus pour l'échec, les niveaux de succès, les rachats de sauvetage et les contrôles."""

import pandas as pd
import pytest

from src.succes import controles_qualite, echec, niveau_succes, rachats_de_sauvetage


@pytest.fixture
def startups():
    return pd.DataFrame(
        {
            "slug": ["morte", "rachetee", "en-bourse", "geante", "grande", "petite"],
            "promotion": ["Winter 2018"] * 6,
            "statut": ["Inactive", "Acquired", "Public", "Active", "Active", "Active"],
            "stage": ["Early", "Early", "Growth", "Growth", "Early", "Early"],
            # Actives de Winter 2018 : 2, 10 et 30 personnes, médiane 10.
            "taille_equipe": [0.0, 3.0, 500.0, 2.0, 30.0, 10.0],
            "region": ["Europe", "Europe", None, "Europe", "Europe", "Europe"],
        }
    )


def test_echec(startups):
    assert echec(startups).tolist() == [1, 0, 0, 0, 0, 0]


def test_niveau_succes_un_cas_par_niveau(startups):
    niveaux = niveau_succes(startups)

    # « geante » est en croissance par son stade, « grande » par son équipe au-dessus de la
    # médiane ; « petite » est exactement à la médiane, donc en survie.
    assert niveaux.tolist() == ["échec", "sortie", "sortie", "croissance", "croissance", "survie"]
    assert niveaux.cat.ordered
    assert niveaux.min() == "échec" and niveaux.max() == "sortie"


def test_niveau_succes_seuil_par_promotion():
    startups = pd.DataFrame(
        {
            "promotion": ["Winter 2018", "Winter 2018", "Summer 2021", "Summer 2021"],
            "statut": ["Active"] * 4,
            "stage": ["Early"] * 4,
            "taille_equipe": [20.0, 40.0, 4.0, 20.0],
        }
    )

    # 20 personnes : sous la médiane de Winter 2018 (30), au-dessus de celle de Summer 2021 (12).
    assert niveau_succes(startups).tolist() == ["survie", "croissance", "survie", "croissance"]


def test_niveau_succes_seuil_fixe(startups):
    niveaux = niveau_succes(startups, seuil_equipe=50)

    assert niveaux[startups["slug"] == "grande"].tolist() == ["survie"]


def test_niveau_succes_statut_inconnu():
    startups = pd.DataFrame(
        {"promotion": ["Winter 2018"], "statut": [None], "stage": ["Early"], "taille_equipe": [5.0]}
    )

    assert niveau_succes(startups).isna().all()


def test_rachats_de_sauvetage(startups):
    sauvetages = rachats_de_sauvetage(startups)

    assert sauvetages.tolist() == [False, True, False, False, False, False]
    assert not rachats_de_sauvetage(startups, taille_max=2).any()


def test_controles_qualite(startups):
    startups = pd.concat([startups, startups.iloc[[5]]], ignore_index=True)
    startups.loc[6, "promotion"] = "Unspecified"

    controles = controles_qualite(startups)

    assert controles.loc["Statut manquant", "nombre"] == 0
    assert controles.loc["Équipe manquante ou à 0", "nombre"] == 1
    assert controles.loc["Slug en double", "nombre"] == 2
    assert controles.loc["Région manquante", "nombre"] == 1
    assert controles.loc["Promotion non reconnue", "nombre"] == 1
    assert controles.loc["Slug en double", "part"] == pytest.approx(2 / 7)
