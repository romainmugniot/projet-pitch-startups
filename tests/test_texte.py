"""Tests unitaires pour le nettoyage des pitchs et les indicateurs textuels."""

import nltk
import numpy as np
import pandas as pd
import pytest

from src.texte import (
    JETON_NOM,
    _benjamini_hochberg,
    indicateurs,
    masquer_nom,
    mots_associes,
    nettoyer_pitch,
    preparer_nltk,
    score_jargon,
    similarite_pitchs,
)


def test_nettoyer_pitch():
    assert nettoyer_pitch("AI-powered Payroll for Doctors!") == "ai powered payroll doctor"
    assert nettoyer_pitch("") == ""
    assert nettoyer_pitch(None) == ""


def test_nettoyer_pitch_garde_chiffres_et_sigles():
    # « b2b » et « 3d » gardent leurs chiffres ; les mots courts ne sont pas lemmatisés.
    assert nettoyer_pitch("B2B SaaS for 3D printing") == "b2b saas 3d printing"
    assert nettoyer_pitch("SMS alerts for iOS") == "sms alert ios"


def test_masquer_nom():
    assert masquer_nom("Termius is an SSH client", "Termius") == f"{JETON_NOM} is an SSH client"
    # Forme courte du nom, sans tenir compte de la casse.
    assert masquer_nom("axiom automates browsers", "Axiom.ai") == f"{JETON_NOM} automates browsers"
    # Un mot qui contient le nom n'est pas touché.
    assert masquer_nom("Fixed prices for repairs", "Fix") == "Fixed prices for repairs"
    assert masquer_nom(None, "Fix") == ""
    assert masquer_nom("Fast repairs", None) == "Fast repairs"
    # Le jeton disparaît au nettoyage.
    assert nettoyer_pitch(masquer_nom("Termius SSH client", "Termius")) == "ssh client"


def test_score_jargon():
    assert score_jargon("AI-powered scalable blockchain platform") == pytest.approx(3 / 4)
    assert score_jargon("Payroll for restaurants") == 0.0
    assert score_jargon("") == 0.0


def test_indicateurs():
    pitchs = pd.Series(
        [
            "The Uber for doctors in 24 hours",
            "AI-powered payroll for restaurants",
            None,
        ],
        index=["a", "b", "c"],
    )
    res = indicateurs(pitchs)

    assert list(res.index) == ["a", "b", "c"]
    assert res.loc["a", "nb_mots"] == 7
    assert bool(res.loc["a", "a_chiffre"]) is True
    assert bool(res.loc["a", "tournure_x_for_y"]) is True
    assert bool(res.loc["a", "client_nomme"]) is True
    assert bool(res.loc["b", "tournure_x_for_y"]) is False
    assert bool(res.loc["b", "client_nomme"]) is True
    assert res.loc["b", "score_jargon"] > 0.0
    assert res.loc["c", "nb_mots"] == 0


@pytest.mark.parametrize(
    "pitch",
    [
        "Copilot for AI teams",
        "Analytics for web apps",
        "Search for data",
        "Tools for every team",
        "Platform for building apps",
    ],
)
def test_fausses_cibles(pitch):
    assert bool(indicateurs(pd.Series([pitch])).loc[0, "client_nomme"]) is False


def test_mots_associes():
    pitchs = pd.Series(
        [
            "Payroll software for clinics",
            "Payroll automation for clinics",
            "Payroll platform for doctors",
            "Crypto metaverse social app",
            "Crypto metaverse gaming app",
            "Crypto social network",
        ]
    )
    succes = pd.Series([1, 1, 1, 0, 0, 0])

    res = mots_associes(pitchs, succes, freq_min=2)
    assert "payroll" in res.index
    assert "crypto" in res.index
    assert res.loc["payroll", "log_odds_ratio"] > 0
    assert res.loc["crypto", "log_odds_ratio"] < 0
    assert res.index[0] in {"payroll", "clinic"}
    assert res.loc["payroll", "z"] > 0 > res.loc["crypto", "z"]
    assert ((res["p_ajustee"] >= res["p_valeur"]) & (res["p_ajustee"] <= 1)).all()


def test_mots_associes_erreur_type():
    # 3 succès sur 3 contre 0 échec sur 3 : log-odds et erreur-type de Woolf calculés à la main.
    pitchs = pd.Series(["payroll"] * 3 + ["crypto"] * 3)
    res = mots_associes(pitchs, pd.Series([1, 1, 1, 0, 0, 0]), freq_min=1)
    log_odds = np.log((3.5 / 0.5) / (0.5 / 3.5))
    erreur_type = np.sqrt(1 / 3.5 + 1 / 0.5 + 1 / 0.5 + 1 / 3.5)
    assert res.loc["payroll", "log_odds_ratio"] == pytest.approx(log_odds)
    assert res.loc["payroll", "z"] == pytest.approx(log_odds / erreur_type)


def test_benjamini_hochberg():
    # Exemple classique : m = 4, p triées 0,01 0,02 0,03 0,5 -> 0,04 0,04 0,04 0,5.
    ajustees = _benjamini_hochberg(np.array([0.03, 0.5, 0.01, 0.02]))
    assert ajustees == pytest.approx([0.04, 0.5, 0.04, 0.04])


def test_similarite_pitchs():
    origine = pd.Series(["Payroll for restaurants", "Cloud storage", ""])
    actuel = pd.Series(["Payroll software for restaurants", "Electric bikes", "Payroll"])
    sim = similarite_pitchs(origine, actuel)

    assert sim.iloc[0] == pytest.approx(2 / 3)
    assert sim.iloc[1] == 0.0
    assert np.isnan(sim.iloc[2])


def test_preparer_nltk_echec_explicite(monkeypatch):
    def introuvable(chemin):
        raise LookupError(chemin)

    monkeypatch.setattr(nltk.data, "find", introuvable)
    monkeypatch.setattr(nltk, "download", lambda paquet, quiet: False)  # comme sans internet

    with pytest.raises(RuntimeError, match="stopwords"):
        preparer_nltk()
