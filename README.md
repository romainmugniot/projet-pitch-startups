# Peut-on prédire le succès d'une startup dès sa première phrase de pitch ?

Projet du cours *Python pour la data science* (Lino Galiana), par Capucine, Gaspard et Romain.

**Lire le rapport en ligne, avec ses graphiques :**
[rapport.ipynb sur nbviewer](https://nbviewer.org/github/romainmugniot/projet-pitch-startups/blob/main/rapport.ipynb)

## La question

Nous cherchons si la phrase de pitch d'une startup Y Combinator, publiée sur Hacker News à son
lancement, dit quelque chose de ce qu'elle est devenue des années plus tard : active, rachetée, en
bourse ou morte.

Deux réponses sont plausibles :

- **Le pitch porte un signal** : les startups au pitch clair et concret (un client précis, un problème
  nommé) survivent mieux que celles au pitch vague ou chargé de buzzwords.
- **Le pitch ne dit rien de plus que le secteur** : une fois la promotion et le secteur pris en compte,
  les mots n'apportent aucune information supplémentaire.

Sous-questions :

1. Quels mots, tournures (« the X for Y », « AI-powered ») ou caractéristiques (longueur, jargon)
   sont associés aux succès et aux échecs ?
2. Le signal tient-il une fois contrôlés le secteur, la promotion et la région ?
3. Les pitchs qui ont fait réagir Hacker News au lancement appartiennent-ils aux startups qui ont réussi ?

## Résultat en bref

Sur 392 startups YC des promotions 2017 à 2021, aucun trait du pitch d'origine (longueur, client
nommé, chiffre, tournure « X for Y », mots à la mode) ni aucun mot n'est associé à l'échec une fois
la promotion et le secteur pris en compte. Le secteur, lui, compte : les startups grand public
échouent 2,2 fois plus qu'attendu. Ni la réaction de Hacker News ni la réécriture du pitch ne
distinguent les destins.

## Données

Deux API publiques et gratuites, **sans clé** : il n'y a aucun secret à gérer.

| Source | Accès | Ce que nous utilisons |
|---|---|---|
| [API Y Combinator (yc-oss)](https://github.com/yc-oss/api) | Fichiers JSON statiques mis à jour chaque jour | Nom, pitch actuel, description, statut, promotion, secteur, tags, taille d'équipe, région, anciens noms |
| [API Hacker News (Algolia)](https://hn.algolia.com/api) | Recherche des posts marqués `launch_hn` | Titre « Launch HN: Nom (YC W20) – pitch », date, points, commentaires |

**Jointure** : le titre de chaque post Launch HN contient le nom de la startup et sa promotion
(« (YC S16) »). Une expression régulière les extrait, puis on les rapproche de la base YC.

**Variable étudiée** : l'échec. Une startup a échoué si la base YC la dit *Inactive* ; actives,
rachetées et en bourse n'ont pas échoué.

**Période** : promotions 2017 à 2021, qui ont au moins cinq ans de recul (392 startups dont le post
Launch HN date de moins d'un an après la promotion).

Les données brutes et nettoyées ne sont **pas** dans le dépôt : des copies datées sont stockées sur le
stockage S3 du SSP Cloud, et le notebook repart de la copie du 30/10/2026 plutôt que des API en direct.
Ces fichiers sont lisibles par tout utilisateur connecté du SSP Cloud :

| Fichier | Contenu |
|---|---|
| `s3://capucine/diffusion/projet-pitch-startups/brut/yc/2026-10-30.parquet` | base Y Combinator brute (6 354 startups) |
| `s3://capucine/diffusion/projet-pitch-startups/brut/hn/2026-10-30.parquet` | posts Launch HN bruts (1 159 posts) |

La table commune (853 startups appariées) est reconstruite à partir de ces deux fichiers à chaque
exécution du notebook.

## Relancer le projet

Le notebook se lance depuis un service VSCode ou Jupyter du [SSP Cloud](https://datalab.sspcloud.fr),
là où sont les données :

```bash
git clone https://github.com/romainmugniot/projet-pitch-startups.git
cd projet-pitch-startups
pip install -e .
```

Puis ouvrir `rapport.ipynb` et exécuter toutes les cellules (*Run All*).

Hors du SSP Cloud, le stockage refuse l'accès : le notebook télécharge alors les données depuis les
API, avec un avertissement, car les chiffres peuvent différer un peu de ceux du rapport.

Configuration (variables d'environnement, déjà définies sur le SSP Cloud) :

| Variable | Rôle | Défaut |
|---|---|---|
| `PITCH_S3_BUCKET` | Bucket S3 où sont lues et écrites les données | `capucine` |
| `AWS_S3_ENDPOINT` | Endpoint S3 du SSP Cloud | `minio.lab.sspcloud.fr` |

Pour repartir des API plutôt que de la copie S3, mettre `COLLECTER = True` dans la section
*Paramètres* du notebook.

## Organisation du dépôt

```
projet-pitch-startups/
├── README.md          # question, sources, comment tout relancer
├── CONTRIBUTING.md    # règles de travail (branches, relecture, notebook)
├── pyproject.toml     # dépendances (installation en une commande)
├── rapport.ipynb      # le rapport final, court et commenté
├── src/
│   ├── config.py      # chemins S3, colonnes de la table commune et leur description
│   ├── collecte.py    # appels aux API YC et Hacker News, sauvegarde S3
│   ├── nettoyage.py   # regex, harmonisation des noms, jointure
│   ├── succes.py      # échec et niveaux de succès
│   ├── texte.py       # nettoyage des pitchs, indicateurs, mots associés à l'échec
│   ├── analyse.py     # taux d'échec et comparaisons
│   └── graphiques.py  # fonctions de visualisation
├── tests/             # tests des fonctions de src/
└── .github/workflows/ # CI : ruff, pytest et exécution du notebook
```

Le notebook importe les fonctions de `src/` au lieu d'accumuler de longues cellules.

## Table commune

Une ligne par startup :

`slug`, `nom`, `promotion`, `pitch_origine`, `pitch_actuel`, `statut`, `niveau_succes`, `secteur`,
`region`, `taille_equipe`, `stage`, `top_company`, `anciens_noms`, `hn_date`, `hn_points`,
`hn_commentaires`, `hn_delai_ans` (années entre le début de la promotion et le post Launch HN).

Leur description est dans `src/config.py` (`DICTIONNAIRE_VARIABLES`) et en annexe A du rapport.

## Contribuer

Règles de travail : voir [CONTRIBUTING.md](CONTRIBUTING.md).

## Limites

Détaillées dans la conclusion du rapport :

- **Recul** : même les promotions 2017 à 2021 perdent encore des startups, et *Active* ne veut pas
  dire réussie.
- **Représentativité** : une startup YC sur cinq seulement publie un Launch HN, surtout en B2B, et
  toutes ont déjà été sélectionnées par YC.
- **Contrôles mesurés aujourd'hui** : le secteur et la région sont ceux de la base YC actuelle.
- **Puissance** : les traits rares (chiffre, « X for Y », mots à la mode) concernent une dizaine
  de startups chacun et ne peuvent être ni confirmés ni écartés.
- **Corrélation et non causalité**, **API YC non officielle** (yc-oss).
