"""Cas attendus pour l'extraction des titres, la normalisation des noms et la jointure."""

import pandas as pd
import pytest

from src.config import COLONNES_TABLE_COMMUNE
from src.nettoyage import (
    bilan_jointure,
    debut_promotion,
    echantillon_etude,
    extraire_titre,
    joindre,
    normaliser_nom,
)


@pytest.mark.parametrize(
    "titre, attendu",
    [
        (
            "Launch HN: Airgoat (YC W20) – Kayak for travel agents",
            {"nom": "Airgoat", "promotion": "Winter 2020", "pitch": "Kayak for travel agents"},
        ),
        (
            "Launch HN: Acme (YC S16) - Payroll for restaurants",
            {"nom": "Acme", "promotion": "Summer 2016", "pitch": "Payroll for restaurants"},
        ),
        (
            "Launch HN: Acme – Payroll for restaurants",
            {"nom": "Acme", "promotion": None, "pitch": "Payroll for restaurants"},
        ),
        (
            "Launch HN: Feather (YC S17), painless furniture rental",
            {"nom": "Feather", "promotion": "Summer 2017", "pitch": "painless furniture rental"},
        ),
        (
            "Launch HN: Tinfoil (YC P25): Verifiable Privacy for Cloud AI",
            {
                "nom": "Tinfoil",
                "promotion": "Spring 2025",
                "pitch": "Verifiable Privacy for Cloud AI",
            },
        ),
        (
            "Launch HN: Prompt Engineering as a Service",
            {"nom": None, "promotion": None, "pitch": None},
        ),
    ],
)
def test_extraire_titre(titre, attendu):
    assert extraire_titre(titre) == attendu


@pytest.mark.parametrize(
    "nom, attendu",
    [
        ("Acme, Inc.", "acme"),
        ("Café Labs", "cafe"),
        ("Acme AI", "acme"),
        ("Acme HQ", "acme"),
        ("Armory.io", "armory"),
        ("Human Layer", "humanlayer"),
        ("Labs", "labs"),
    ],
)
def test_normaliser_nom(nom, attendu):
    assert normaliser_nom(nom) == attendu


@pytest.fixture
def yc():
    return pd.DataFrame(
        {
            "slug": ["airgoat", "muzz"],
            "name": ["Airgoat Inc.", "Muzz"],
            "batch": ["Winter 2020", "Summer 2017"],
            "one_liner": ["Travel software", "Muslim dating app"],
            "status": ["Active", "Acquired"],
            "industry": ["B2B", "Consumer"],
            "regions": [["United States of America", "America / Canada"], ["Europe"]],
            "team_size": [12.0, 50.0],
            "former_names": [[], ["muzmatch"]],
            "stage": ["Early", "Growth"],
            "top_company": [False, True],
        }
    )


@pytest.fixture
def hn():
    return pd.DataFrame(
        {
            "objectID": ["1", "2", "3", "4"],
            "title": [
                "Launch HN: Airgoat (YC W20) – Kayak for travel agents",
                "Launch HN: Airgoat (YC W20) – Second launch",
                "Launch HN: Muzmatch (YC S17) – Matchmaking for Muslims",
                "Launch HN: Acme – Payroll for restaurants",
            ],
            "created_at": [
                "2020-01-10T10:00:00Z",
                "2021-05-01T10:00:00Z",
                "2017-07-01T10:00:00Z",
                "2018-01-01T10:00:00Z",
            ],
            "created_at_i": [1578650400, 1619863200, 1498903200, 1514800800],
            "points": [100, 20, 80, 5],
            "num_comments": [40, 3, 30, 1],
        }
    )


def test_joindre(yc, hn):
    table = joindre(yc, hn).set_index("slug")

    assert list(table.reset_index().columns) == COLONNES_TABLE_COMMUNE
    # Deux posts pour Airgoat : on garde le plus ancien. Muzz est retrouvée par son ancien nom.
    assert table.loc["airgoat", "pitch_origine"] == "Kayak for travel agents"
    assert table.loc["muzz", "pitch_origine"] == "Matchmaking for Muslims"
    assert table.loc["airgoat", "region"] == "America / Canada"
    assert table.loc["muzz", "stage"] == "Growth"
    assert bool(table.loc["muzz", "top_company"])
    # Post du 10 janvier 2020, promotion Winter 2020 commencée le 1er janvier : 9 jours.
    assert table.loc["airgoat", "hn_delai_ans"] == pytest.approx(9 / 365.25)
    assert len(table) == 2


@pytest.mark.parametrize(
    "promotion, attendu",
    [
        ("Winter 2020", pd.Timestamp("2020-01-01")),
        ("Spring 2025", pd.Timestamp("2025-04-01")),
        ("Summer 2017", pd.Timestamp("2017-06-01")),
        ("Fall 2024", pd.Timestamp("2024-09-01")),
    ],
)
def test_debut_promotion(promotion, attendu):
    assert debut_promotion(promotion) == attendu


def test_debut_promotion_inconnue():
    assert pd.isna(debut_promotion("Unspecified"))


def test_bilan_jointure(yc, hn):
    bilan = bilan_jointure(joindre(yc, hn), hn)

    assert bilan.index.tolist() == ["Summer 2017", "Winter 2020", "inconnue"]
    assert bilan["posts"].tolist() == [1, 2, 1]
    assert bilan["appariements"].tolist() == [1, 1, 0]


def test_echantillon_etude():
    table = pd.DataFrame(
        {
            "slug": ["a", "b", "c", "d", "e"],
            "promotion": ["Winter 2016", "Summer 2017", "Fall 2021", "Winter 2022", "Winter 2019"],
            "statut": ["Inactive", "Inactive", "Active", "Inactive", "Acquired"],
            "hn_delai_ans": [0.1, 0.2, 0.5, 0.1, 3.0],
        }
    )

    echantillon, etapes = echantillon_etude(table)

    # 2016 et 2022 sont hors période ; « e » a posté trois ans après sa promotion.
    assert echantillon["slug"].tolist() == ["b", "c"]
    assert etapes["startups"].tolist() == [5, 3, 2]
    assert etapes["echecs"].tolist() == [3, 1, 1]
    assert etapes["part_echecs"].iloc[-1] == 0.5


def test_echantillon_etude_autres_bornes():
    table = pd.DataFrame(
        {
            "promotion": ["Winter 2016", "Summer 2017"],
            "statut": ["Inactive", "Active"],
            "hn_delai_ans": [0.1, 1.5],
        }
    )

    echantillon, etapes = echantillon_etude(table, debut=2016, fin=2017, delai_max=2.0)

    assert len(echantillon) == 2
    assert etapes.index[-1] == "Post Launch HN au plus 2 ans après la promotion"
