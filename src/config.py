"""Configuration partagée : emplacements S3 et colonnes de la table commune."""

import os

# Bucket S3 du SSP Cloud où sont stockées les données (ex. "prenom-nom" ou "projet-pitch").
S3_BUCKET = os.environ.get("PITCH_S3_BUCKET", "capucine")
S3_ENDPOINT = "https://" + os.environ.get("AWS_S3_ENDPOINT", "minio.lab.sspcloud.fr")
# Sous « diffusion/ », les fichiers sont lisibles par les autres utilisateurs du SSP Cloud.
S3_PREFIX = f"{S3_BUCKET}/diffusion/projet-pitch-startups"

CHEMIN_BRUT = f"{S3_PREFIX}/brut"
CHEMIN_PROPRE = f"{S3_PREFIX}/propre"

URL_YC = "https://yc-oss.github.io/api/companies/all.json"
URL_HN = "https://hn.algolia.com/api/v1/search_by_date"

# Colonnes de la table commune, fixées en semaine 1 : chacun code en visant ces noms.
COLONNES_TABLE_COMMUNE = [
    "slug",
    "nom",
    "promotion",
    "pitch_origine",
    "pitch_actuel",
    "statut",
    "niveau_succes",
    "secteur",
    "region",
    "taille_equipe",
    "anciens_noms",
    "hn_date",
    "hn_points",
    "hn_commentaires",
]
