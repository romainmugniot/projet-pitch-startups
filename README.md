# Peut-on prédire le succès d'une startup dès sa première phrase de pitch ?

Projet du cours *Python pour la data science* (Lino Galiana), par Capucine, Gaspard et Romain.

## La question

Nous testons si la phrase de pitch d'une startup Y Combinator, écrite le jour de son lancement,
permet de prédire ce qu'elle est devenue des années plus tard : active, rachetée, en bourse ou morte.

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

## Données

Deux API publiques et gratuites, **sans clé** : il n'y a aucun secret à gérer.

| Source | Accès | Ce que nous utilisons |
|---|---|---|
| [API Y Combinator (yc-oss)](https://github.com/yc-oss/api) | Fichiers JSON statiques mis à jour chaque jour | Nom, pitch actuel, description, statut, promotion, secteur, tags, taille d'équipe, région, anciens noms |
| [API Hacker News (Algolia)](https://hn.algolia.com/api) | Recherche des posts marqués `launch_hn` | Titre « Launch HN: Nom (YC W20) – pitch », date, points, commentaires |

**Jointure** : le titre de chaque post Launch HN contient le nom de la startup et sa promotion
(« (YC S16) »). Une expression régulière les extrait, puis on les rapproche de la base YC.

**Variable à prédire** : succès si la startup est active, rachetée ou en bourse ; échec si elle est inactive.

Les données brutes et nettoyées ne sont **pas** dans le dépôt : des copies datées sont stockées sur le
stockage S3 du SSP Cloud, et le notebook repart de la copie du 30/10/2026 plutôt que des API en direct.
Ces fichiers sont lisibles par tout utilisateur connecté du SSP Cloud :

| Fichier | Contenu |
|---|---|
| `s3://capucine/diffusion/projet-pitch-startups/brut/yc/2026-10-30.parquet` | base Y Combinator brute (6 354 startups) |
| `s3://capucine/diffusion/projet-pitch-startups/brut/hn/2026-10-30.parquet` | posts Launch HN bruts (1 159 posts) |
| `s3://capucine/diffusion/projet-pitch-startups/propre/table_commune/2026-10-30.parquet` | table commune (853 startups) |

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

Pour reconstruire les données depuis les API (au lieu des copies S3), voir la section *Collecte* du notebook.

## Organisation du dépôt

```
projet-pitch-startups/
├── README.md          # question, sources, comment tout relancer
├── pyproject.toml     # dépendances (installation en une commande)
├── rapport.ipynb      # le rapport final, court et commenté
├── src/
│   ├── config.py      # chemins S3, colonnes de la table commune
│   ├── collecte.py    # appels aux API YC et Hacker News, sauvegarde S3
│   ├── nettoyage.py   # regex, harmonisation des noms, jointure
│   ├── succes.py      # niveaux de succès
│   ├── texte.py       # nettoyage des pitchs, indicateurs, score de jargon
│   ├── analyse.py     # statistiques et modèles
│   └── graphiques.py  # fonctions de visualisation
└── tests/             # tests des fonctions de nettoyage
```

Le notebook importe les fonctions de `src/` au lieu d'accumuler de longues cellules.

## Table commune

Une ligne par startup :

`slug`, `nom`, `promotion`, `pitch_origine`, `pitch_actuel`, `statut`, `niveau_succes`, `secteur`,
`region`, `taille_equipe`, `anciens_noms`, `hn_date`, `hn_points`, `hn_commentaires`.

Le dictionnaire des variables se trouve en annexe du rapport.

## Contribuer

Règles de travail : voir [CONTRIBUTING.md](CONTRIBUTING.md).

## Limites anticipées

- **Pitch actuel contre pitch d'origine** : les survivantes réécrivent leur pitch, d'où l'usage du pitch Launch HN.
- **Recul contre couverture** : les posts Launch HN sont surtout fréquents pour les promotions récentes.
- **« Active » ne veut pas dire réussie** : la taille d'équipe sert de mesure complémentaire.
- **Échantillon déjà sélectionné** par YC, **corrélation et non causalité**, **API non officielle**.
