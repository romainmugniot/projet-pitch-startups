"""Tests unitaires pour le nettoyage des pitchs et les indicateurs textuels."""

import nltk
import numpy as np
import pandas as pd
import pytest

from src.texte import (
    indicateurs,
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
