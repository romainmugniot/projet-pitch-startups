"""Cas attendus pour l'extraction des titres, la normalisation des noms et la jointure."""

import pandas as pd
import pytest

from src.config import COLONNES_TABLE_COMMUNE
from src.nettoyage import bilan_jointure, extraire_titre, joindre, normaliser_nom


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
    assert len(table) == 2


def test_bilan_jointure(yc, hn):
    bilan = bilan_jointure(joindre(yc, hn), hn)

    assert bilan.index.tolist() == ["Summer 2017", "Winter 2020", "inconnue"]
    assert bilan["posts"].tolist() == [1, 2, 1]
    assert bilan["appariements"].tolist() == [1, 1, 0]
