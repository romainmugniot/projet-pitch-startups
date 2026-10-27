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
