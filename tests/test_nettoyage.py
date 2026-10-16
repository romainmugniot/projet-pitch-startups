"""Cas attendus pour l'extraction des titres et la normalisation des noms.

Ils sont marqués `xfail` tant que les fonctions ne sont pas écrites : retirer le marqueur
dans la pull request qui les implémente.
"""

import pytest

from src.nettoyage import extraire_titre, normaliser_nom

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="à implémenter")


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
    ],
)
def test_extraire_titre(titre, attendu):
    assert extraire_titre(titre) == attendu


@pytest.mark.parametrize(
    "nom, attendu",
    [("Acme, Inc.", "acme"), ("Café Labs", "cafe"), ("Acme AI", "acme"), ("Acme HQ", "acme")],
)
def test_normaliser_nom(nom, attendu):
    assert normaliser_nom(nom) == attendu
