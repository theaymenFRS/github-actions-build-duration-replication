# data/

Ce répertoire contient les jeux de données nécessaires pour comprendre et reproduire les expériences du mémoire.

- `project_links.csv` liste les 10 projets sélectionnés avec leurs URL GitHub publiques ainsi que les fichiers bruts et traités correspondants.
- `raw/` contient les fichiers CSV au niveau projet issus de la collecte des données GitHub Actions.
- `processed/` contient les fichiers CSV spécifiques aux workflows réellement utilisés dans les expériences, après correction et filtrage.

Pour le prétraitement appliqué dans le code des expériences (filtres, lags, etc.), voir `scripts/preprocessing/README.md` et `scripts/preprocessing/preprocess_common.py`.

## Remarques

- `raw/` correspond aux fichiers du dossier d’origine **SELECTED PROJECTS**.
- `processed/` correspond aux fichiers du dossier d’origine **project_combine**.
