"""Configuration partagée : emplacements S3 et colonnes de la table commune."""

import os

# Bucket S3 du SSP Cloud où sont stockées les données (ex. "prenom-nom" ou "projet-pitch").
S3_BUCKET = os.environ.get("PITCH_S3_BUCKET", "capucine")
S3_ENDPOINT = "https://" + os.environ.get("AWS_S3_ENDPOINT", "minio.lab.sspcloud.fr")
# Sous « diffusion/ », les fichiers sont lisibles par les autres utilisateurs du SSP Cloud.
S3_PREFIX = f"{S3_BUCKET}/diffusion/projet-pitch-startups"

CHEMIN_BRUT = f"{S3_PREFIX}/brut"

# Copie des données analysée dans le rapport : le notebook lit toujours celle-ci, même si une
# collecte plus récente est déposée à côté sur le S3.
DATE_COPIE = "2026-10-30"

URL_YC = "https://yc-oss.github.io/api/companies/all.json"
URL_HN = "https://hn.algolia.com/api/v1/search_by_date"

# Colonnes de la table commune, fixées en semaine 1 : chacun code en visant ces noms. Le dictionnaire
# donne aussi leur description, affichée en annexe du rapport.
DICTIONNAIRE_VARIABLES = {
    "slug": "Identifiant unique de la startup dans la base YC",
    "nom": "Nom actuel de la startup dans la base YC",
    "promotion": "Promotion YC (« Winter 2020 », « Summer 2021 »…)",
    "pitch_origine": "Pitch du post Launch HN, publié au lancement",
    "pitch_actuel": "Pitch actuel de la base YC (champ one_liner), réécrit au fil du temps",
    "statut": "Statut actuel dans la base YC : Active, Inactive (échec), Acquired ou Public",
    "niveau_succes": "Échec, survie, croissance ou sortie (succes.niveau_succes), calculé après "
    "la jointure",
    "secteur": "Secteur principal dans la base YC (B2B, Consumer, Fintech, Healthcare…)",
    "region": "Grande région de la base YC (America / Canada, Europe…), vide si aucune",
    "taille_equipe": "Nombre de salariés indiqué aujourd'hui dans la base YC",
    "stage": "Stade actuel dans la base YC : Early ou Growth",
    "top_company": "Vrai si YC classe la startup parmi ses « top companies »",
    "anciens_noms": "Liste des anciens noms de la startup",
    "hn_date": "Date et heure (UTC) du post Launch HN",
    "hn_points": "Points (votes) reçus par le post Launch HN",
    "hn_commentaires": "Nombre de commentaires du post Launch HN",
    "hn_delai_ans": "Années écoulées entre le début de la promotion et le post Launch HN",
}
COLONNES_TABLE_COMMUNE = list(DICTIONNAIRE_VARIABLES)
