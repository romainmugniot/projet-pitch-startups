"""Les graphiques tracent les bons chiffres, dans le bon ordre, avec la mise en forme commune."""

import pandas as pd
import pytest

from src.analyse import taux_echec
from src.graphiques import (
    SOURCE,
    barres_taux,
    courbe_mots_par_annee,
    courbe_recul,
    mots_frequence_association,
    points_ratios,
)


def test_barres_taux():
    startups = pd.DataFrame(
        {
            "statut": ["Inactive", "Inactive", "Active", "Inactive", "Active", "Active", "Active"],
            "secteur": ["Consumer", "Consumer", "Consumer", "B2B", "B2B", "B2B", "B2B"],
        }
    )
    taux = taux_echec(startups, "secteur", min_effectif=1)

    fig = barres_taux(taux, "Les startups grand public échouent plus souvent")

    barres = fig.data[0]
    assert barres.x.tolist() == pytest.approx([2 / 3, 1 / 4])
    assert barres.y.tolist() == ["Consumer (3)", "B2B (4)"]
    # Le taux le plus haut est affiché en haut : dernier de l'ordre des catégories de l'axe.
    assert fig.layout.yaxis.categoryarray == ("B2B (4)", "Consumer (3)")
    assert barres.error_x.array.tolist() == pytest.approx((taux["ic_haut"] - taux["taux"]).tolist())
    assert fig.layout.shapes[0].x0 == pytest.approx(3 / 7)  # taux de l'ensemble
    assert fig.layout.title.text == "Les startups grand public échouent plus souvent"
    assert fig.layout.annotations[-1].text == SOURCE


def test_courbe_recul():
    yc = pd.DataFrame(
        {
            "batch": ["Winter 2017"] * 10 + ["Summer 2025"] * 10 + ["Winter 2026"] * 2,
            "status": ["Inactive"] * 3 + ["Active"] * 7 + ["Inactive"] + ["Active"] * 11,
        }
    )
    table = pd.DataFrame(
        {"promotion": ["Winter 2017"] * 10, "statut": ["Inactive"] * 4 + ["Active"] * 6}
    )

    fig = courbe_recul(yc, table)

    base_yc, launch_hn = fig.data
    # 2026 n'a que 2 startups, sous le minimum de 10 : l'année n'est pas tracée.
    assert base_yc.x.tolist() == [2017, 2025]
    assert base_yc.y.tolist() == pytest.approx([0.3, 0.1])
    assert launch_hn.x.tolist() == [2017]
    assert launch_hn.y.tolist() == pytest.approx([0.4])
    bande = fig.layout.shapes[0]
    assert (bande.x0, bande.x1) == (2016.5, 2021.5)


def test_courbe_mots_par_annee():
    parts = pd.DataFrame({"ai": [0.1, 0.3], "api": [0.05, 0.04]}, index=[2022, 2023])

    fig = courbe_mots_par_annee(parts, "« AI » envahit les pitchs")

    assert [trace.name for trace in fig.data] == ["ai", "api"]
    assert fig.data[0].x.tolist() == [2022, 2023]
    assert fig.data[0].y.tolist() == pytest.approx([0.1, 0.3])


def test_points_ratios():
    ratios = pd.DataFrame(
        {
            "effectif": [10, 40],
            "observes": [4, 8],
            "attendus": [2.0, 10.0],
            "ratio": [2.0, 0.8],
            "ic_bas": [0.5, 0.35],
            "ic_haut": [5.1, 1.6],
        },
        index=pd.Index(["Chiffre", "Client nommé"], name="indicateur"),
    )

    fig = points_ratios(ratios, "Aucun indicateur ne s'écarte du hasard")

    points = fig.data[0]
    assert points.x.tolist() == pytest.approx([2.0, 0.8])
    assert points.y.tolist() == ["Chiffre (10)", "Client nommé (40)"]
    assert points.error_x.arrayminus.tolist() == pytest.approx([1.5, 0.45])
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.shapes[0].x0 == 1


def test_mots_frequence_association():
    mots = pd.DataFrame(
        {
            "frequence": [20, 15, 30],
            "nb_succes": [19, 3, 22],
            "nb_echec": [1, 12, 8],
            "log_odds_ratio": [1.5, -2.0, 0.1],
            "z": [3.0, -4.0, 0.2],
            "p_ajustee": [0.01, 0.02, 0.9],
        },
        index=pd.Index(["software", "crypto", "app"], name="mot"),
    )

    fig = mots_frequence_association(mots, "Peu de mots se détachent", nb_etiquettes=2)

    groupes = {trace.name: trace for trace in fig.data}
    assert set(groupes) == {
        "Associé au succès",
        "Associé à l'échec",
        "Non significatif après correction",
    }
    assert groupes["Associé à l'échec"].customdata[0][0] == "crypto"
    # Seuls les deux mots au plus grand |z| sont nommés.
    assert groupes["Non significatif après correction"].text.tolist() == [""]


def test_mots_frequence_association_mots_au_meme_point():
    # « tool » et « find » sont vus dans les mêmes pitchs : un seul point, nommé par les deux mots.
    mots = pd.DataFrame(
        {
            "frequence": [7, 7, 10],
            "nb_succes": [7, 7, 5],
            "nb_echec": [0, 0, 5],
            "log_odds_ratio": [1.5, 1.5, -0.2],
            "z": [1.0, 1.0, -0.3],
            "p_ajustee": [0.8, 0.8, 0.9],
        },
        index=pd.Index(["tool", "find", "data"], name="mot"),
    )

    fig = mots_frequence_association(mots, "Aucun mot ne se détache", nb_etiquettes=2)

    points = fig.data[0]
    assert len(points.x) == 2
    assert points.text.tolist() == ["tool, find", ""]
