"""Tests unitaires pour le découpage temporel et les pipelines de modélisation."""

import pandas as pd
import pytest

from src.analyse import decoupage_temporel, evaluer, modele_reference, modele_texte


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
