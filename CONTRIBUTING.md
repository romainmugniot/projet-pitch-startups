# Règles de travail

Le barème pénalise les dépôts où le projet est simplement déposé à la fin : l'historique Git doit
montrer le travail de chacun.

## Une branche par tâche

```bash
git switch main
git pull
git switch -c collecte-launch-hn      # nom court qui décrit la tâche
# ... travail, commits fréquents ...
git push -u origin collecte-launch-hn
```

Puis ouvrir une pull request sur GitHub. **On ne pousse jamais directement sur `main`.**

## Relecture croisée

Chaque pull request est relue et approuvée par un autre membre avant d'être fusionnée :

| Auteur | Relecteur |
|---|---|
| Membre 1 (données) | Membre 2 |
| Membre 2 (texte et modèle) | Membre 3 |
| Membre 3 (succès et rapport) | Membre 1 |

## Le notebook

- `rapport.ipynb` a **un seul propriétaire : le membre 3**. Les deux autres contribuent par des
  fonctions dans `src/`, que le notebook importe. Cela évite les conflits Git sur le notebook.
- Pas de longues cellules ni de copier-coller : une cellule appelle une fonction, puis on commente le résultat.
- Avant de committer le notebook, vider les sorties (`nbstripout` est dans les dépendances `dev`) :

```bash
pip install -e ".[dev]"
nbstripout --install
```

## Code

- Chaque fonction a une docstring qui dit ce qu'elle prend et ce qu'elle renvoie.
- Les colonnes produites suivent les noms de la table commune (`src/config.py`).
- Pas de données dans le dépôt : tout passe par le S3 du SSP Cloud.
- Avant d'ouvrir une pull request :

```bash
ruff check src tests
pytest
```

## Point d'équipe

30 minutes par semaine : avancement, blocages, prochaines tâches.
