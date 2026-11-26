"""Sauvegarde et relecture S3, testées sur un stockage en mémoire à la place du vrai S3."""

import datetime

import fsspec
import pandas as pd
import pytest

from src import collecte


@pytest.fixture
def s3_memoire(monkeypatch):
    stockage = fsspec.filesystem("memory")
    stockage.store.clear()
    monkeypatch.setattr(collecte, "connexion_s3", lambda: stockage)
    return stockage


def test_sauvegarder_puis_lire(s3_memoire):
    df = pd.DataFrame({"nom": ["Airgoat", "Acme"], "anciens_noms": [["Goat"], []]})

    chemin = collecte.sauvegarder_s3(df, "yc", "bucket/brut")

    assert chemin == f"bucket/brut/yc/{datetime.date.today().isoformat()}.parquet"
    relu = collecte.lire_s3("yc", "bucket/brut")
    assert relu["nom"].tolist() == ["Airgoat", "Acme"]
    assert [list(noms) for noms in relu["anciens_noms"]] == [["Goat"], []]


def test_lire_la_plus_recente_ou_une_date(s3_memoire):
    for jour, valeur in [("2026-09-01", 1), ("2026-10-15", 2)]:
        with s3_memoire.open(f"bucket/brut/hn/{jour}.parquet", "wb") as fichier:
            pd.DataFrame({"valeur": [valeur]}).to_parquet(fichier)

    assert collecte.lire_s3("hn", "bucket/brut")["valeur"].tolist() == [2]
    assert collecte.lire_s3("hn", "bucket/brut", date="2026-09-01")["valeur"].tolist() == [1]


def test_lire_sans_copie(s3_memoire):
    with pytest.raises(FileNotFoundError):
        collecte.lire_s3("inconnu", "bucket/brut")


def test_charger_brut_lit_la_copie_du_rapport(monkeypatch):
    lectures = []

    def lire_s3(nom, dossier, date):
        lectures.append((nom, dossier, date))
        return pd.DataFrame({"slug": ["acme"]})

    monkeypatch.setattr(collecte, "lire_s3", lire_s3)
    monkeypatch.setattr(collecte, "telecharger_yc", lambda: pytest.fail("API appelée"))

    assert collecte.charger_brut("yc")["slug"].tolist() == ["acme"]
    assert lectures == [("yc", collecte.CHEMIN_BRUT, collecte.DATE_COPIE)]


def test_charger_brut_repli_sur_l_api(monkeypatch):
    def lire_s3_refuse(nom, dossier, date):
        raise PermissionError("Forbidden")

    monkeypatch.setattr(collecte, "lire_s3", lire_s3_refuse)
    monkeypatch.setattr(collecte, "telecharger_launch_hn", lambda: pd.DataFrame({"points": [42]}))

    with pytest.warns(UserWarning, match="peuvent différer"):
        posts = collecte.charger_brut("hn")
    assert posts["points"].tolist() == [42]


def test_charger_brut_collecte_demandee(monkeypatch):
    monkeypatch.setattr(collecte, "lire_s3", lambda *args: pytest.fail("S3 lu"))
    monkeypatch.setattr(collecte, "telecharger_yc", lambda: pd.DataFrame({"slug": ["neuve"]}))

    with pytest.warns(UserWarning, match="collecte demandée"):
        assert collecte.charger_brut("yc", collecter=True)["slug"].tolist() == ["neuve"]
