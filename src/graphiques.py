"""Graphiques homogènes (Plotly Express) : titre, axes nommés, source, mêmes couleurs partout."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import DATE_COPIE
from src.nettoyage import debut_promotion
from src.succes import echec

# Le bleu et le rouge restent distincts pour les daltoniens, contrairement au vert et au rouge.
# « neutre » sert aux repères (base YC complète, taux de l'ensemble) et aux textes secondaires.
COULEURS = {"succes": "#257de0", "echec": "#ec4647", "neutre": "#666666"}

SOURCE = (
    "Sources : base Y Combinator (yc-oss) et posts Launch HN (Algolia), "
    f"copie du {pd.Timestamp(DATE_COPIE):%d/%m/%Y}."
)


def _mise_en_forme(
    fig: go.Figure, titre: str, axe_x: str | None, axe_y: str | None, source: str = SOURCE
) -> go.Figure:
    """Même présentation pour tous les graphiques du rapport.

    Le `titre` doit dire le résultat (« Les startups grand public échouent plus souvent »), pas
    seulement le sujet ; `axe_x` et `axe_y` nomment les axes avec leur unité (None : pas de
    titre d'axe) ; `source` s'affiche en note sous le graphique. Fond blanc, grille discrète,
    texte gris foncé.
    """
    fig.update_layout(
        template="plotly_white",
        title={"text": titre, "x": 0, "xanchor": "left"},
        xaxis_title=axe_x,
        yaxis_title=axe_y,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": "#191919"},
        margin={"l": 10, "r": 30, "t": 80, "b": 90},
        hoverlabel={"bgcolor": "white"},
    )
    fig.update_xaxes(gridcolor="#e7e7e7", linecolor="#c2c2c2", zeroline=False)
    fig.update_yaxes(gridcolor="#e7e7e7", linecolor="#c2c2c2", zeroline=False)
    fig.add_annotation(
        text=source,
        xref="paper",
        yref="paper",
        x=0,
        y=0,
        yshift=-70,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font={"size": 11, "color": COULEURS["neutre"]},
    )
    return fig


def courbe_recul(
    yc: pd.DataFrame,
    table: pd.DataFrame,
    debut: int = 2017,
    fin: int = 2021,
    min_effectif: int = 10,
    titre: str = "Les promotions récentes n'ont pas encore eu le temps d'échouer",
) -> go.Figure:
    """Part de startups inactives par année de promotion, dans la base YC et dans la table commune.

    `yc` est la base YC brute (colonnes `batch`, `status`), `table` la table commune. Plus une
    promotion est récente, moins ses startups ont eu le temps d'échouer : la part d'inactives
    s'effondre pour les dernières années, quelle que soit la qualité des startups. La bande grise
    marque les promotions étudiées (`debut` à `fin`). Les années de moins de `min_effectif`
    startups ne sont pas tracées.
    """
    series = {
        "Toutes les startups YC": yc.rename(columns={"batch": "promotion", "status": "statut"}),
        "Startups avec un Launch HN": table,
    }
    morceaux = []
    for nom, df in series.items():
        annee = df["promotion"].map(debut_promotion).dt.year.rename("annee")
        par_annee = echec(df).groupby(annee).agg(effectif="size", part="mean").reset_index()
        morceaux.append(par_annee[par_annee["effectif"] >= min_effectif].assign(serie=nom))
    donnees = pd.concat(morceaux, ignore_index=True)
    donnees["annee"] = donnees["annee"].astype(int)

    fig = px.line(
        donnees,
        x="annee",
        y="part",
        color="serie",
        markers=True,
        custom_data=["effectif"],
        color_discrete_map={
            "Toutes les startups YC": COULEURS["neutre"],
            "Startups avec un Launch HN": COULEURS["echec"],
        },
    )
    fig.update_traces(
        line_width=2,
        marker_size=8,
        hovertemplate="Promotions %{x} : %{y:.0%} d'inactives (%{customdata[0]} startups)",
    )
    fig.add_vrect(
        x0=debut - 0.5,
        x1=fin + 0.5,
        fillcolor="#efefef",
        line_width=0,
        layer="below",
        annotation_text=f"Promotions étudiées ({debut}-{fin})",
        annotation_position="top left",
        annotation_font_color=COULEURS["neutre"],
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(dtick=2)
    fig.update_layout(
        legend={"title": None, "orientation": "h", "x": 0, "y": 1, "yanchor": "bottom"}
    )
    return _mise_en_forme(fig, titre, "Année de la promotion", "Part de startups inactives")


def barres_taux(taux: pd.DataFrame, titre: str, source: str = SOURCE) -> go.Figure:
    """Barres horizontales du taux d'échec par modalité, avec l'intervalle de confiance à 95 %.

    `taux` est le tableau renvoyé par `analyse.taux_echec`, dont l'ordre est gardé (du taux le
    plus haut, en haut, au plus bas). L'effectif est rappelé dans l'étiquette (« Consumer (62) ») :
    un taux sur 12 startups est bien moins sûr qu'un taux sur 200, ce que montre aussi la largeur
    de l'intervalle. La ligne grise marque le taux d'échec de l'ensemble.
    """
    donnees = taux.reset_index()
    modalite = donnees.columns[0]
    donnees["etiquette"] = (
        donnees[modalite].astype(str) + " (" + donnees["effectif"].astype(str) + ")"
    )
    donnees["erreur_haut"] = donnees["ic_haut"] - donnees["taux"]
    donnees["erreur_bas"] = donnees["taux"] - donnees["ic_bas"]

    fig = px.bar(
        donnees,
        x="taux",
        y="etiquette",
        orientation="h",
        error_x="erreur_haut",
        error_x_minus="erreur_bas",
        custom_data=["echecs", "effectif", "ic_bas", "ic_haut"],
        color_discrete_sequence=[COULEURS["echec"]],
    )
    fig.update_traces(
        error_x={"color": COULEURS["neutre"], "thickness": 1.5, "width": 4},
        hovertemplate=(
            "%{y} : %{x:.0%} d'échecs (%{customdata[0]} sur %{customdata[1]})<br>"
            "Intervalle à 95 % : %{customdata[2]:.0%} à %{customdata[3]:.0%}<extra></extra>"
        ),
    )
    ensemble = taux["echecs"].sum() / taux["effectif"].sum()
    fig.add_vline(
        x=ensemble,
        line_width=1,
        line_color=COULEURS["neutre"],
        annotation_text=f"Ensemble : {ensemble:.0%}",
        annotation_position="top",
        annotation_font_color=COULEURS["neutre"],
    )
    fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    fig.update_yaxes(categoryorder="array", categoryarray=donnees["etiquette"].tolist()[::-1])
    fig.update_layout(bargap=0.45, barcornerradius=4, height=160 + 40 * len(donnees))
    return _mise_en_forme(fig, titre, "Taux d'échec (startups inactives)", None, source)


def mots_frequence_association(mots: pd.DataFrame):
    """Nuage de points : fréquence d'un mot contre son association au succès."""
    raise NotImplementedError
