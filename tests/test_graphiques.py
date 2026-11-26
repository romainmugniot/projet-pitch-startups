"""Les graphiques tracent les bons chiffres, dans le bon ordre, avec la mise en forme commune."""

import pandas as pd
import pytest

from src.analyse import taux_echec
from src.graphiques import SOURCE, barres_taux, courbe_recul


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
