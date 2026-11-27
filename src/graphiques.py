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


def courbe_mots_par_annee(parts: pd.DataFrame, titre: str) -> go.Figure:
    """Part des pitchs qui contiennent chaque mot, par année de promotion : une ligne par mot.

    `parts` a une ligne par année (index) et une colonne par mot, avec des parts entre 0 et 1.
    """
    donnees = (
        parts.rename_axis("annee").reset_index().melt("annee", var_name="mot", value_name="part")
    )
    fig = px.line(
        donnees,
        x="annee",
        y="part",
        color="mot",
        markers=True,
        color_discrete_sequence=[
            COULEURS["succes"],
            COULEURS["echec"],
            "#d9a400",
            COULEURS["neutre"],
        ],
    )
    fig.update_traces(
        line_width=2,
        marker_size=7,
        hovertemplate="« %{fullData.name} » : %{y:.0%} des pitchs des promotions %{x}<extra></extra>",
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(dtick=1)
    fig.update_layout(
        legend={"title": None, "orientation": "h", "x": 0, "y": 1, "yanchor": "bottom"}
    )
    return _mise_en_forme(
        fig, titre, "Année de la promotion", "Part des pitchs qui contiennent le mot"
    )


def points_ratios(ratios: pd.DataFrame, titre: str, source: str = SOURCE) -> go.Figure:
    """Ratios échecs observés / attendus avec leur intervalle à 95 %, un point par ligne.

    `ratios` est indexé par l'étiquette à afficher et a les colonnes `effectif`, `observes`,
    `attendus`, `ratio`, `ic_bas` et `ic_haut` (`analyse.ratios_par_indicateur`,
    `analyse.taux_echec_standardise`). L'axe est logarithmique pour qu'un ratio de 2 et un ratio
    de 0,5 soient à la même distance de la ligne verticale à 1 (« autant d'échecs qu'attendu »).
    Un intervalle qui coupe cette ligne signale un écart compatible avec le hasard.
    """
    donnees = ratios.reset_index()
    etiquette = donnees.columns[0]
    donnees["etiquette"] = (
        donnees[etiquette].astype(str) + " (" + donnees["effectif"].astype(str) + ")"
    )
    donnees["erreur_haut"] = donnees["ic_haut"] - donnees["ratio"]
    donnees["erreur_bas"] = donnees["ratio"] - donnees["ic_bas"]

    fig = px.scatter(
        donnees,
        x="ratio",
        y="etiquette",
        error_x="erreur_haut",
        error_x_minus="erreur_bas",
        log_x=True,
        custom_data=["observes", "attendus", "ic_bas", "ic_haut"],
        color_discrete_sequence=[COULEURS["echec"]],
    )
    fig.update_traces(
        marker_size=10,
        error_x={"color": COULEURS["neutre"], "thickness": 1.5, "width": 4},
        hovertemplate=(
            "%{y} : %{customdata[0]} échecs pour %{customdata[1]:.1f} attendus "
            "(ratio %{x:.2f})<br>Intervalle à 95 % : %{customdata[2]:.2f} à %{customdata[3]:.2f}"
            "<extra></extra>"
        ),
    )
    fig.add_vline(
        x=1,
        line_width=1,
        line_color=COULEURS["neutre"],
        annotation_text="Autant d'échecs qu'attendu",
        annotation_position="top",
        annotation_font_color=COULEURS["neutre"],
    )
    fig.update_yaxes(categoryorder="array", categoryarray=donnees["etiquette"].tolist()[::-1])
    # Graduations lisibles sur l'axe logarithmique, symétriques autour de 1.
    graduations = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    fig.update_xaxes(tickvals=graduations, ticktext=["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    fig.update_layout(height=160 + 45 * len(donnees))
    return _mise_en_forme(
        fig, titre, "Échecs observés / attendus à promotion égale (échelle log)", None, source
    )


def mots_frequence_association(
    mots: pd.DataFrame, titre: str, seuil: float = 0.05, nb_etiquettes: int = 10
) -> go.Figure:
    """Nuage de points : fréquence d'un mot contre son association au succès.

    `mots` est le tableau de `texte.mots_associes`. En ordonnée, le log-odds ratio : au-dessus de 0,
    le mot est plus fréquent chez les startups qui n'ont pas échoué. Les mots dont la `p_ajustee`
    est sous `seuil` sont colorés (bleu côté succès, rouge côté échec), les autres restent gris ;
    les `nb_etiquettes` mots les plus nettement associés (plus grand |z|) sont nommés. Plusieurs mots
    vus dans les mêmes pitchs tombent sur le même point : ils sont nommés ensemble
    (« tool, find »).
    """
    donnees = mots.reset_index()
    significatif = donnees["p_ajustee"] < seuil
    donnees["groupe"] = "Non significatif après correction"
    donnees.loc[significatif & (donnees["log_odds_ratio"] > 0), "groupe"] = "Associé au succès"
    donnees.loc[significatif & (donnees["log_odds_ratio"] < 0), "groupe"] = "Associé à l'échec"
    nommes = donnees["mot"].isin(
        donnees.loc[donnees["z"].abs().nlargest(nb_etiquettes).index, "mot"]
    )
    position = ["frequence", "nb_succes", "nb_echec"]
    noms = donnees[nommes].groupby(position)["mot"].agg(", ".join).rename("etiquette")
    tous = donnees.groupby(position)["mot"].agg(", ".join).rename("mots_point")
    donnees = (
        donnees.drop_duplicates(subset=position + ["groupe"])
        .merge(noms, on=position, how="left")
        .merge(tous, on=position)
    )
    donnees["etiquette"] = donnees["etiquette"].fillna("")

    fig = px.scatter(
        donnees,
        x="frequence",
        y="log_odds_ratio",
        color="groupe",
        text="etiquette",
        custom_data=["mots_point", "nb_succes", "nb_echec", "p_ajustee"],
        color_discrete_map={
            "Non significatif après correction": COULEURS["neutre"],
            "Associé au succès": COULEURS["succes"],
            "Associé à l'échec": COULEURS["echec"],
        },
    )
    fig.update_traces(
        marker={"size": 9, "opacity": 0.8},
        textposition="top center",
        textfont={"size": 11, "color": "#191919"},
        hovertemplate=(
            "« %{customdata[0]} » : %{customdata[1]} startups sans échec, %{customdata[2]} échecs"
            "<br>Log-odds : %{y:.2f} ; p ajustée : %{customdata[3]:.2f}<extra></extra>"
        ),
    )
    fig.add_hline(y=0, line_width=1, line_color=COULEURS["neutre"])
    fig.update_layout(
        legend={"title": None, "orientation": "h", "x": 0, "y": 1, "yanchor": "bottom"}
    )
    return _mise_en_forme(
        fig,
        titre,
        "Nombre de pitchs qui contiennent le mot",
        "Log-odds ratio (> 0 : côté succès)",
    )


def boites_reaction(df: pd.DataFrame, titre: str, colonne: str = "hn_points") -> go.Figure:
    """Boîtes à moustaches des points du post Launch HN (ou de `colonne`) par statut.

    Échelle logarithmique : les points vont de quelques unités à plus de 1 000, et une échelle
    linéaire écraserait les boîtes. Les inactives sont en rouge, les autres statuts en bleu.
    """
    ordre = ["Acquired", "Active", "Inactive"]
    statuts = [s for s in ordre if s in set(df["statut"])]
    fig = px.box(
        df[df["statut"].isin(statuts)],
        x="statut",
        y=colonne,
        color="statut",
        log_y=True,
        points="outliers",
        category_orders={"statut": statuts},
        color_discrete_map={
            "Acquired": COULEURS["succes"],
            "Active": COULEURS["succes"],
            "Inactive": COULEURS["echec"],
        },
    )
    fig.update_layout(showlegend=False)
    graduations = [1, 3, 10, 30, 100, 300, 1000, 3000]
    fig.update_yaxes(
        tickvals=graduations, ticktext=[f"{g:,}".replace(",", " ") for g in graduations]
    )
    noms = {"hn_points": "Points du post Launch HN", "hn_commentaires": "Commentaires du post"}
    return _mise_en_forme(fig, titre, None, f"{noms.get(colonne, colonne)} (échelle log)")
