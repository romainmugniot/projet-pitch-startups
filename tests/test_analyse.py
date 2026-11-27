"""Tests unitaires : taux d'échec, découpage temporel et pipelines de modélisation."""

import pandas as pd
import pytest

from src.analyse import (
    decoupage_temporel,
    evaluer,
    modele_reference,
    modele_texte,
    ratios_par_indicateur,
    reaction_hn_et_destin,
    taux_echec,
    taux_echec_standardise,
)


@pytest.fixture
def deux_promotions():
    # Promotion A : 2 échecs sur 4 (50 %) ; promotion B : 1 échec sur 4 (25 %).
    return pd.DataFrame(
        {
            "promotion": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "secteur": ["X", "X", "Y", "Y", "X", "Y", "Y", "Y"],
            "statut": [
                "Inactive",
                "Inactive",
                "Active",
                "Active",
                "Active",
                "Active",
                "Inactive",
                "Acquired",
            ],
        }
    )


def test_taux_echec(deux_promotions):
    taux = taux_echec(deux_promotions, "secteur", min_effectif=1)

    # X : 2 échecs sur 3 ; Y : 1 échec sur 5. Trié du taux le plus haut au plus bas.
    assert taux.index.tolist() == ["X", "Y"]
    assert taux["effectif"].tolist() == [3, 5]
    assert taux["echecs"].tolist() == [2, 1]
    assert taux["taux"].tolist() == pytest.approx([2 / 3, 1 / 5])
    assert (taux["ic_bas"] < taux["taux"]).all() and (taux["taux"] < taux["ic_haut"]).all()


def test_taux_echec_intervalle_de_wilson():
    startups = pd.DataFrame({"statut": ["Inactive"] * 2 + ["Active"] * 8, "secteur": "B2B"})

    # Valeurs de référence de l'intervalle de Wilson pour 2 succès sur 10 : [0,057 ; 0,510].
    ligne = taux_echec(startups, "secteur").loc["B2B"]
    assert ligne["ic_bas"] == pytest.approx(0.0567, abs=1e-4)
    assert ligne["ic_haut"] == pytest.approx(0.5098, abs=1e-4)


def test_taux_echec_regroupe_les_modalites_rares():
    startups = pd.DataFrame(
        {
            "statut": ["Inactive"] * 3 + ["Active"] * 10,
            "region": ["Europe"] * 10 + ["Africa", None, None],
        }
    )

    assert taux_echec(startups, "region").index.tolist() == ["Europe", "Autres"]
    taux = taux_echec(startups, "region", min_effectif=2)
    assert taux.index.tolist() == ["Europe", "Autres", "Non renseigné"]
    assert taux["effectif"].tolist() == [10, 1, 2]
    assert taux.loc["Non renseigné", "ic_bas"] == 0


def test_taux_echec_standardise(deux_promotions):
    ratios = taux_echec_standardise(deux_promotions, "secteur", min_effectif=1)

    # Attendus : chaque startup compte pour le taux de sa promotion (A : 0,5 ; B : 0,25).
    # X : 0,5 + 0,5 + 0,25 = 1,25 échec attendu pour 2 observés ; Y : 0,5 × 2 + 0,25 × 3 = 1,75
    # pour 1 observé.
    assert ratios["observes"].tolist() == [2, 1]
    assert ratios["attendus"].tolist() == pytest.approx([1.25, 1.75])
    assert ratios["ratio"].tolist() == pytest.approx([1.6, 4 / 7])
    # Au total, autant d'échecs attendus qu'observés.
    assert ratios["attendus"].sum() == pytest.approx(ratios["observes"].sum())


def test_taux_echec_standardise_intervalle():
    startups = pd.DataFrame(
        {"promotion": "A", "statut": ["Inactive"] * 10 + ["Active"] * 10, "secteur": "B2B"}
    )

    # Une seule promotion : 10 échecs attendus et 10 observés, ratio 1. Intervalle de Poisson
    # exact pour 10 événements : [4,80 ; 18,39], que l'approximation de Byar retrouve.
    ligne = taux_echec_standardise(startups, "secteur").loc["B2B"]
    assert ligne["ratio"] == pytest.approx(1)
    assert ligne["ic_bas"] == pytest.approx(0.480, abs=2e-3)
    assert ligne["ic_haut"] == pytest.approx(1.839, abs=2e-3)


def test_ratios_par_indicateur(deux_promotions):
    startups = deux_promotions.assign(
        chiffre=[1, 1, 0, 0, 1, 0, 0, 0],  # 0/1 accepté comme booléen
        jamais=False,
    )

    ratios = ratios_par_indicateur(startups, ["chiffre", "jamais"])

    # Indicateur vrai pour 3 startups (2 de A, 1 de B) : 2 échecs pour 0,5 + 0,5 + 0,25 attendus.
    # « jamais » n'est vrai pour personne : pas de ligne.
    assert ratios.index.tolist() == ["chiffre"]
    assert ratios.loc["chiffre", "effectif"] == 3
    assert ratios.loc["chiffre", "observes"] == 2
    assert ratios.loc["chiffre", "ratio"] == pytest.approx(2 / 1.25)


def test_reaction_hn_et_destin():
    startups = pd.DataFrame(
        {
            "statut": ["Inactive"] * 3 + ["Acquired"] * 2,
            "hn_points": [10, 20, 30, 100, 200],
            "hn_commentaires": [1, 2, 3, 4, 5],
        }
    )

    reaction = reaction_hn_et_destin(startups)

    assert reaction.index.tolist() == ["Acquired", "Inactive"]
    assert reaction["startups"].tolist() == [2, 3]
    assert reaction.loc["Inactive", "points_mediane"] == 20
    assert reaction.loc["Inactive", "points_q1"] == 15
    assert reaction.loc["Acquired", "commentaires_q3"] == pytest.approx(4.75)


@pytest.fixture
def table_exemple():
    return pd.DataFrame(
        {
            "promotion": [
                "Winter 2018",
                "Winter 2018",
                "Summer 2018",
                "Summer 2018",
                "Winter 2019",
                "Winter 2019",
                "Summer 2020",
                "Summer 2020",
                "Winter 2021",
                "Winter 2021",
            ],
            "secteur": [
                "B2B",
                "Consumer",
                "B2B",
                "Consumer",
                "Fintech",
                "Consumer",
                "B2B",
                "Consumer",
                "Fintech",
                "Consumer",
            ],
            "region": [
                "America / Canada",
                "Europe",
                "America / Canada",
                "Europe",
                "America / Canada",
                None,
                "America / Canada",
                "Europe",
                "America / Canada",
                "Europe",
            ],
            "pitch_origine": [
                "Payroll software for restaurants",
                "Crypto social app for friends",
                "Billing automation for clinics",
                "Crypto gaming metaverse app",
                "Accounting API for startups",
                "Social photo sharing app",
                "Payroll automation for clinics",
                "Crypto social gaming app",
                "Billing API for restaurants",
                "Social metaverse app",
            ],
            "statut": [
                "Active",
                "Inactive",
                "Acquired",
                "Inactive",
                "Active",
                "Inactive",
                "Active",
                "Inactive",
                "Active",
                "Inactive",
            ],
        }
    )


def test_decoupage_temporel(table_exemple):
    X_train, X_test, y_train, y_test = decoupage_temporel(
        table_exemple, promotions_test=["Summer 2020", "Winter 2021"]
    )
    assert len(X_train) == 6
    assert len(X_test) == 4
    assert set(X_test["promotion"]) == {"Summer 2020", "Winter 2021"}
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_pipelines_et_evaluation(table_exemple):
    X_train, X_test, y_train, y_test = decoupage_temporel(
        table_exemple, promotions_test=["Summer 2020", "Winter 2021"]
    )

    ref = modele_reference().fit(X_train, y_train)
    txt = modele_texte(min_df=1).fit(X_train, y_train)

    metriques_ref = evaluer(ref, X_test, y_test)
    metriques_txt = evaluer(txt, X_test, y_test)

    cles_attendues = {
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "f1_echec",
        "f1_succes",
        "matrice_confusion",
    }
    assert set(metriques_ref.keys()) == cles_attendues
    assert set(metriques_txt.keys()) == cles_attendues
    assert 0.0 <= metriques_ref["roc_auc"] <= 1.0
    assert 0.0 <= metriques_txt["roc_auc"] <= 1.0
    assert metriques_txt["matrice_confusion"].shape == (2, 2)
