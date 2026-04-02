# data/

Ce répertoire contient les jeux de données nécessaires pour comprendre et reproduire les expériences du mémoire.

- `project_links.csv` liste les 10 projets sélectionnés avec leurs URL GitHub publiques ainsi que les fichiers bruts et traités correspondants.
- `raw/` contient les fichiers CSV spécifiques aux workflows réellement utilisés dans les expériences (collecté a partir de ghaminer en filtrant par workflow id)
- `processed/` contient les fichiers CSV après filtrage.

Pour le prétraitement appliqué dans le code des expériences (filtres, lags, etc.), voir `scripts/preprocessing/README.md` et `scripts/preprocessing/preprocess_common.py`.