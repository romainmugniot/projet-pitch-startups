"""Analyse : statistiques descriptives et modèles."""

from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.succes import echec
from src.texte import nettoyer_pitch

COLONNES_STRUCTURELLES = ["secteur", "promotion", "region"]

# Quantile de la loi normale pour les intervalles de confiance à 95 % (1,96).
Z_95 = NormalDist().inv_cdf(0.975)

AUTRES = "Autres"
NON_RENSEIGNE = "Non renseigné"


def _modalites(df: pd.DataFrame, par: str, min_effectif: int) -> pd.Series:
    """Valeurs de la colonne `par`, prêtes à comparer.

    Les valeurs manquantes deviennent « Non renseigné » et les modalités de moins de
    `min_effectif` startups sont regroupées en « Autres » : un taux calculé sur trois startups
    ne dit presque rien.
    """
    valeurs = df[par].astype("object").fillna(NON_RENSEIGNE)
    effectifs = valeurs.map(valeurs.value_counts())
    return valeurs.where(effectifs >= min_effectif, AUTRES)


def _trier(resultat: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Trie du plus grand au plus petit sur `colonne`, « Autres » et « Non renseigné » en dernier."""
    resultat = resultat.sort_values(colonne, ascending=False)
    en_dernier = resultat.index.isin([AUTRES, NON_RENSEIGNE])
    return pd.concat([resultat[~en_dernier], resultat[en_dernier]])


def _intervalle_wilson(
    nombre: pd.Series, effectif: pd.Series, z: float = Z_95
) -> tuple[pd.Series, pd.Series]:
    """Bornes de l'intervalle de Wilson (95 % par défaut) de la proportion `nombre / effectif`.

    Plus fiable que « taux ± 1,96 écart-type » sur de petits effectifs ou des taux proches de 0 :
    ses bornes restent toujours entre 0 et 1.
    """
    p = nombre / effectif
    centre = (p + z**2 / (2 * effectif)) / (1 + z**2 / effectif)
    demi_largeur = (
        z / (1 + z**2 / effectif) * np.sqrt(p * (1 - p) / effectif + z**2 / (4 * effectif**2))
    )
    return (centre - demi_largeur).clip(lower=0), (centre + demi_largeur).clip(upper=1)


def _intervalle_poisson(observes: pd.Series, z: float = Z_95) -> tuple[pd.Series, pd.Series]:
    """Bornes de l'intervalle à 95 % d'un nombre d'échecs supposé suivre une loi de Poisson.

    Approximation de Byar : les nombres d'échecs « vrais » compatibles avec les `observes` échecs
    comptés. Précise dès quelques échecs ; pour 0 échec, la borne basse vaut 0.
    """
    o = observes.astype(float)
    bas = o * (1 - 1 / (9 * o) - z / (3 * np.sqrt(o))) ** 3
    haut = (o + 1) * (1 - 1 / (9 * (o + 1)) + z / (3 * np.sqrt(o + 1))) ** 3
    return bas.where(o > 0, 0.0), haut


def taux_echec(df: pd.DataFrame, par: str, min_effectif: int = 10) -> pd.DataFrame:
    """Taux d'échec par modalité de `par` (secteur, région…), avec son intervalle de confiance.

    Une ligne par modalité : `effectif`, `echecs` (au sens de `succes.echec`), `taux`, et les
    bornes `ic_bas` et `ic_haut` de l'intervalle de Wilson à 95 %. Les modalités de moins de
    `min_effectif` startups sont regroupées en « Autres », les valeurs manquantes en
    « Non renseigné ». Lignes triées du taux le plus haut au plus bas, ces deux groupes en dernier.
    """
    groupes = _modalites(df, par, min_effectif)
    resultat = echec(df).groupby(groupes).agg(effectif="size", echecs="sum")
    resultat["taux"] = resultat["echecs"] / resultat["effectif"]
    resultat["ic_bas"], resultat["ic_haut"] = _intervalle_wilson(
        resultat["echecs"], resultat["effectif"]
    )
    return _trier(resultat, "taux")


def taux_echec_standardise(df: pd.DataFrame, par: str, min_effectif: int = 10) -> pd.DataFrame:
    """Échecs observés rapportés aux échecs attendus à promotion égale, par modalité de `par`.

    Une promotion récente a eu moins de temps pour échouer : comparer des taux bruts reviendrait
    en partie à comparer des âges. Chaque startup reçoit donc le taux d'échec de sa promotion ;
    leur somme sur une modalité donne les échecs « attendus » si ses startups échouaient comme
    leurs promotions (standardisation indirecte).

    Une ligne par modalité (regroupées comme dans `taux_echec`) : `effectif`, `observes`,
    `attendus`, `ratio` = observés / attendus, et son intervalle à 95 % (`ic_bas`, `ic_haut`),
    en supposant que le nombre d'échecs suit une loi de Poisson. Un ratio de 1,3 se lit
    « 30 % d'échecs de plus que leurs promotions », 0,8 « 20 % de moins » ; si l'intervalle
    contient 1, l'écart peut venir du hasard.
    """
    echecs = echec(df)
    attendus = echecs.groupby(df["promotion"]).transform("mean")
    groupes = _modalites(df, par, min_effectif)
    resultat = (
        pd.DataFrame({"echec": echecs, "attendu": attendus})
        .groupby(groupes)
        .agg(effectif=("echec", "size"), observes=("echec", "sum"), attendus=("attendu", "sum"))
    )
    resultat["ratio"] = resultat["observes"] / resultat["attendus"]
    bas, haut = _intervalle_poisson(resultat["observes"])
    resultat["ic_bas"] = bas / resultat["attendus"]
    resultat["ic_haut"] = haut / resultat["attendus"]
    return _trier(resultat, "ratio")


def reaction_hn_et_destin(df: pd.DataFrame) -> pd.DataFrame:
    """Lien entre la réaction sur Hacker News (points, commentaires) et le destin de la startup."""
    raise NotImplementedError


def decoupage_temporel(
    df: pd.DataFrame,
    promotions_test: list[str],
    cible: str = "succes",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Entraînement sur les promotions anciennes, test sur les plus récentes.

    Si `cible` est absente de `df`, elle est construite depuis `statut` (1 si statut != "Inactive",
    0 sinon). Renvoie `(X_train, X_test, y_train, y_test)`.
    """
    masque_test = df["promotion"].isin(promotions_test)
    if masque_test.all() or (~masque_test).all():
        raise ValueError(
            "Le découpage temporel doit contenir au moins une observation en train et en test."
        )

    if cible in df.columns:
        y = df[cible].astype(int)
        X = df.drop(columns=[cible])
    else:
        y = (df["statut"] != "Inactive").astype(int)
        X = df.copy()

    return (
        X.loc[~masque_test].reset_index(drop=True),
        X.loc[masque_test].reset_index(drop=True),
        y.loc[~masque_test].reset_index(drop=True),
        y.loc[masque_test].reset_index(drop=True),
    )


def _serie_en_texte(serie: pd.Series) -> pd.Series:
    """Remplace les valeurs manquantes d'une colonne de texte par des chaînes vides."""
    return serie.fillna("").astype(str)


def _encodeur_structurel() -> Pipeline:
    """Imputation des modalités manquantes puis encodage one-hot robuste aux modalités inédites."""
    return Pipeline(
        [
            ("imputation", SimpleImputer(strategy="constant", fill_value="Inconnu")),
            ("encodage", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def modele_reference() -> Pipeline:
    """Pipeline scikit-learn : régression logistique sur secteur, promotion et région."""
    pretraitement = ColumnTransformer(
        transformers=[
            ("structure", _encodeur_structurel(), COLONNES_STRUCTURELLES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pretraitement", pretraitement),
            ("modele", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )


def modele_texte(
    colonne_pitch: str = "pitch_origine",
    min_df: int = 2,
    max_features: int = 1000,
) -> Pipeline:
    """Même pipeline que `modele_reference`, plus le pitch transformé en TF-IDF."""
    branche_texte = Pipeline(
        [
            ("nettoyage_na", FunctionTransformer(_serie_en_texte, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=nettoyer_pitch,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=max_features,
                ),
            ),
        ]
    )
    pretraitement = ColumnTransformer(
        transformers=[
            ("structure", _encodeur_structurel(), COLONNES_STRUCTURELLES),
            ("texte", branche_texte, colonne_pitch),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pretraitement", pretraitement),
            ("modele", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )


def evaluer(modele: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """AUC, PR-AUC, balanced accuracy, F1 par classe et matrice de confusion.

    Tient compte du déséquilibre succès / échec.
    """
    y_pred = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_echec": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "f1_succes": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "matrice_confusion": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }
